//! WAV input and output, resampling, and silence trimming.
use crate::gpu::R;

/// Mono samples in [-1, 1) and the sample rate. Channels are averaged (librosa's mono=True).
pub fn read_wav(path: &str) -> R<(Vec<f32>, usize)> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{path}: not a RIFF/WAVE file").into());
    }
    let u16le = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32le = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    let (mut fmt, mut ch, mut sr, mut bits) = (0u16, 0usize, 0usize, 0u16);
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let n = u32le(i + 4) as usize;
        let body = i + 8;
        if body + n > b.len() && id != b"data" {
            return Err(format!("{path}: truncated {} chunk", String::from_utf8_lossy(id)).into());
        }
        if id == b"fmt " {
            if n < 16 || (u16le(body) == 0xfffe && n < 26) {
                return Err(format!("{path}: short fmt chunk").into());
            }
            fmt = u16le(body);
            ch = u16le(body + 2) as usize;
            sr = u32le(body + 4) as usize;
            bits = u16le(body + 14);
            if fmt == 0xfffe {
                fmt = u16le(body + 24); // WAVE_FORMAT_EXTENSIBLE: sub-format GUID starts with the format code
            }
        } else if id == b"data" {
            if ch == 0 || bits == 0 {
                return Err(format!("{path}: data before fmt").into());
            }
            if !matches!((fmt, bits), (1, 16) | (1, 24) | (1, 32) | (3, 32)) {
                return Err(format!("{path}: unsupported WAV format {fmt}/{bits} bit (16/24/32-bit PCM or 32-bit float)").into());
            }
            let frame = ch * bits as usize / 8;
            // a streaming writer leaves the size at 0xFFFFFFFF: then the data runs to the end of the file
            let streamed = n == 0xffff_ffff;
            if !streamed && (body + n > b.len() || !n.is_multiple_of(frame)) {
                return Err(format!("{path}: data chunk is truncated or not whole frames").into());
            }
            let data = if streamed { &b[body..] } else { &b[body..body + n] };
            let nf = data.len() / frame;
            let sample = |k: usize| -> f32 {
                let o = k * bits as usize / 8;
                match (fmt, bits) {
                    (1, 16) => i16::from_le_bytes([data[o], data[o + 1]]) as f32 / 32768.0,
                    (1, 24) => (i32::from_le_bytes([0, data[o], data[o + 1], data[o + 2]]) >> 8) as f32 / 8_388_608.0,
                    (1, 32) => i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0,
                    _ => f32::from_le_bytes(data[o..o + 4].try_into().unwrap()),
                }
            };
            let y: Vec<f32> = (0..nf).map(|t| (0..ch).map(|c| sample(t * ch + c)).sum::<f32>() / ch as f32).collect();
            if y.iter().any(|v| !v.is_finite()) {
                return Err(format!("{path}: samples that are not finite numbers").into());
            }
            return Ok((y, sr));
        }
        i = body + n + (n & 1);
    }
    Err(format!("{path}: no data chunk").into())
}

fn bessel_i0(x: f64) -> f64 {
    let (mut s, mut t, mut k) = (1.0, 1.0, 1.0);
    while t > 1e-12 * s {
        t *= (x / (2.0 * k)).powi(2);
        s += t;
        k += 1.0;
    }
    s
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Polyphase Kaiser-windowed sinc resampling shaped like soxr's "HQ" setting, which librosa uses (passband 91.3% of
/// the lower Nyquist frequency, stopband 100%, beta 13.5); within 1e-4 of librosa on speech.
pub fn resample(y: &[f32], sr: usize, to: usize) -> R<Vec<f32>> {
    if sr == to {
        return Ok(y.to_vec());
    }
    if !(1000..=768_000).contains(&sr) || !(1000..=768_000).contains(&to) {
        return Err(format!("unsupported sample rate {sr} -> {to}").into());
    }
    let g = gcd(sr, to);
    let (up, down) = (to / g, sr / g); // output k sits at input position k * down / up
    let nyq = (to as f64 / sr as f64).min(1.0) / 2.0; // lower Nyquist frequency, cycles per input sample
    let (pass, stop, beta) = (0.913, 1.0, 13.5);
    let fc = (pass + stop) / 2.0 * nyq;
    let atten = beta / 0.1102 + 8.7; // dB, inverse of Kaiser's beta formula
    let half = ((atten - 7.95) / (2.285 * 2.0 * std::f64::consts::PI * (stop - pass) * nyq) / 2.0).ceil() as isize;
    let i0b = bessel_i0(beta);
    let taps = |ph: usize| -> Vec<f64> {
        let frac = ph as f64 / up as f64;
        let taps: Vec<f64> = (-half..=half + 1)
            .map(|n| {
                let t = n as f64 - frac;
                let sinc = if t == 0.0 { 2.0 * fc } else { (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t) };
                let r = t / half as f64;
                if r.abs() > 1.0 { 0.0 } else { sinc * bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b }
            })
            .collect();
        let norm: f64 = taps.iter().sum();
        taps.into_iter().map(|w| w / norm).collect()
    };
    // the filters of every phase are tabulated unless the ratio is awkward enough to make that table large
    let table: Option<Vec<Vec<f64>>> = (up * (2 * half as usize + 2) <= 1 << 22).then(|| (0..up).map(taps).collect());
    let n_out = (y.len() * up).div_ceil(down);
    let one = |k: usize| -> f32 {
        let pos = k * down;
        let (c, ph) = ((pos / up) as isize, pos % up);
        let computed;
        let w = match &table {
            Some(t) => &t[ph],
            None => {
                computed = taps(ph);
                &computed
            }
        };
        let mut acc = 0.0f64;
        for (n, &w) in w.iter().enumerate() {
            let idx = c + n as isize - half;
            if idx >= 0 && (idx as usize) < y.len() {
                acc += w * y[idx as usize] as f64;
            }
        }
        acc as f32
    };
    let mut out = vec![0f32; n_out];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(n_out.div_ceil(4096)).max(1);
    let chunk = n_out.div_ceil(threads);
    std::thread::scope(|s| {
        for (i, part) in out.chunks_mut(chunk.max(1)).enumerate() {
            s.spawn(move || {
                for (j, v) in part.iter_mut().enumerate() {
                    *v = one(i * chunk + j);
                }
            });
        }
    });
    Ok(out)
}

/// Writes mono audio as a WAV file: 16-bit PCM (scaled by 32767 and clipped, as libsndfile does) or 32-bit float.
pub fn write_wav(path: &str, y: &[f32], sr: usize, float: bool) -> R<()> {
    let (fmt, bits) = if float { (3u16, 32u16) } else { (1, 16) };
    let data_len = y.len() * bits as usize / 8;
    if data_len > u32::MAX as usize - 36 {
        return Err(format!("{path}: audio too long for a WAV file").into());
    }
    let mut b = Vec::with_capacity(44 + data_len);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&fmt.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&(sr as u32).to_le_bytes());
    b.extend_from_slice(&(sr as u32 * bits as u32 / 8).to_le_bytes());
    b.extend_from_slice(&(bits / 8).to_le_bytes());
    b.extend_from_slice(&bits.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &v in y {
        if float {
            b.extend_from_slice(&v.to_le_bytes());
        } else {
            b.extend_from_slice(&((v * 32767.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes());
        }
    }
    std::fs::write(path, b).map_err(|e| format!("{path}: {e}").into())
}

/// Cuts leading and trailing silence: keeps from `pad` seconds before the first sample above 2% of the peak to
/// `pad` seconds after the last one.
pub fn trim(y: &[f32], sr: usize, pad: f64) -> Vec<f32> {
    let peak = y.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let loud = |v: &f32| v.abs() > 0.02 * peak;
    let (Some(first), Some(last)) = (y.iter().position(loud), y.iter().rposition(loud)) else { return y.to_vec() };
    let p = (pad * sr as f64) as usize;
    y[first.saturating_sub(p)..(last + p).min(y.len())].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(fmt: u16, ch: u16, sr: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&fmt.to_le_bytes());
        b.extend_from_slice(&ch.to_le_bytes());
        b.extend_from_slice(&sr.to_le_bytes());
        b.extend_from_slice(&(sr * ch as u32 * bits as u32 / 8).to_le_bytes());
        b.extend_from_slice(&(ch * bits / 8).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    fn read(bytes: &[u8]) -> R<(Vec<f32>, usize)> {
        let p = std::env::temp_dir().join(format!("ve-wav-{}-{}.wav", std::process::id(), bytes.len()));
        std::fs::write(&p, bytes).unwrap();
        let r = read_wav(p.to_str().unwrap());
        std::fs::remove_file(&p).unwrap();
        r
    }

    #[test]
    fn wav_formats() {
        let pcm16: Vec<u8> = [16384i16, -32768, 0, 8192].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (y, sr) = read(&wav(1, 2, 48000, 16, &pcm16)).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(y, vec![-0.25, 0.125]); // stereo frames averaged
        let f32le: Vec<u8> = [0.5f32, -1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(read(&wav(3, 1, 16000, 32, &f32le)).unwrap().0, vec![0.5, -1.0]);
        let pcm24 = [0x00, 0x00, 0x40, 0x00, 0x00, 0xc0];
        assert_eq!(read(&wav(1, 1, 16000, 24, &pcm24)).unwrap().0, vec![0.5, -0.5]);
    }

    #[test]
    fn wav_round_trip_and_trim() {
        let p = std::env::temp_dir().join(format!("ve-out-{}.wav", std::process::id()));
        let p = p.to_str().unwrap();
        let y = [0.0f32, 0.5, -1.0, 1.5];
        write_wav(p, &y, 48000, false).unwrap();
        let (r, sr) = read_wav(p).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(r, vec![0.0, 16384.0 / 32768.0, -32767.0 / 32768.0, 32767.0 / 32768.0]);
        write_wav(p, &y, 16000, true).unwrap();
        assert_eq!(read_wav(p).unwrap().0, y.to_vec());
        std::fs::remove_file(p).unwrap();
        let mut z = vec![0f32; 100];
        z[40] = 1.0;
        z[60] = -0.5;
        assert_eq!(trim(&z, 100, 0.05).len(), 60 + 5 - (40 - 5));
        assert_eq!(trim(&[0.0; 4], 100, 0.05).len(), 4);
    }

    #[test]
    fn wav_malformed_is_an_error() {
        assert!(read(b"RIFF\0\0\0\0WAVE").is_err()); // no chunks
        assert!(read(&wav(1, 1, 16000, 8, &[0, 1])).is_err()); // 8-bit is not supported
        assert!(read(&wav(1, 1, 16000, 16, &[]).as_slice()[..30]).is_err()); // fmt chunk cut short
        assert!(read(b"not a wav file").is_err());
        let mut cut = wav(1, 1, 16000, 16, &[0, 1, 2, 3]);
        cut.truncate(cut.len() - 1);
        assert!(read(&cut).is_err()); // data chunk shorter than its size
        assert!(read(&wav(1, 1, 16000, 16, &[0, 1, 2])).is_err()); // half a frame
        let mut streamed = wav(1, 1, 16000, 16, &[0, 0x40, 0, 0x40]);
        streamed[40..44].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        assert_eq!(read(&streamed).unwrap().0, vec![0.5, 0.5]); // size left unset by a streaming writer
    }

    fn tone(sr: usize, hz: f64, secs: f64) -> Vec<f32> {
        (0..(sr as f64 * secs) as usize).map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / sr as f64).sin() as f32).collect()
    }

    fn rms(y: &[f32]) -> f64 {
        let m = &y[y.len() / 4..3 * y.len() / 4]; // away from the edges
        (m.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / m.len() as f64).sqrt()
    }

    #[test]
    fn resampler_keeps_the_band_and_removes_aliases() {
        for sr in [8000, 22050, 44100, 48000] {
            let y = resample(&vec![1.0; sr], sr, 16000).unwrap();
            assert_eq!(y.len(), 16000);
            assert!((y[8000] - 1.0).abs() < 1e-4, "{sr}: DC gain {}", y[8000]);
            let pass = rms(&resample(&tone(sr, 1000.0, 1.0), sr, 16000).unwrap());
            assert!((pass - 0.5f64.sqrt()).abs() < 0.01, "{sr}: 1 kHz rms {pass}");
            if sr > 16000 {
                let stop = rms(&resample(&tone(sr, 9000.0, 1.0), sr, 16000).unwrap());
                assert!(stop < 1e-3, "{sr}: 9 kHz leaks {stop}");
            }
        }
        assert!(resample(&[0.0; 10], 100, 16000).is_err());
        // 767,999 -> 768,000 Hz has 768,000 phases: filters computed per sample instead of tabulated
        assert_eq!(resample(&[0.0; 100], 767_999, 768_000).unwrap().len(), 101);
    }
}
