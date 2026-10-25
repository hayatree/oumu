//! Reader for `torch.save` checkpoints (a zip of `data.pkl` plus one raw storage per tensor) that never runs code
//! from the file: the pickle interpreter knows only dicts, tensors and storages, like `torch.load(weights_only=True)`.
//! Only stored (uncompressed) zip entries are supported, which is what `torch.save` writes.
use std::collections::HashMap;

type R<T> = Result<T, Box<dyn std::error::Error>>;

const MAX_STACK: usize = 1 << 20;
/// Bounds on what a pickle may build: nesting depth, and values and string bytes created by copies (memo, DUP),
/// so that a small file cannot expand into an enormous object.
const MAX_DEPTH: usize = 32;
const MAX_COPIED: usize = 1 << 22;
const MAX_ELEMENTS: usize = 1 << 28;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorRef {
    pub dtype: DType,
    pub key: String,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

#[derive(Clone, Debug)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float, // parsed and dropped: no float is needed
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String, String),
    Storage(DType, String),
    Tensor(TensorRef),
    Mark,
}

impl V {
    fn children(&self) -> &[V] {
        match self {
            V::Tuple(v) | V::List(v) => v,
            _ => &[],
        }
    }
    /// Values in the tree plus the bytes of its strings, counting at most up to `limit`.
    fn size(&self, limit: usize) -> usize {
        let mut n = 1 + if let V::Str(s) | V::Storage(_, s) | V::Global(_, s) = self { s.len() } else { 0 };
        let mut each = |v: &V| {
            if n <= limit {
                n += v.size(limit - n);
            }
        };
        match self {
            V::Dict(d) => d.iter().for_each(|(k, v)| {
                each(k);
                each(v);
            }),
            v => v.children().iter().for_each(each),
        }
        n
    }
    fn deeper_than(&self, depth: usize) -> bool {
        depth == 0
            || match self {
                V::Dict(d) => d.iter().any(|(k, v)| k.deeper_than(depth - 1) || v.deeper_than(depth - 1)),
                v => v.children().iter().any(|c| c.deeper_than(depth - 1)),
            }
    }
}

fn storage_dtype(name: &str) -> Option<DType> {
    Some(match name {
        "FloatStorage" => DType::F32,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::BF16,
        "IntStorage" => DType::I32,
        "LongStorage" => DType::I64,
        _ => return None,
    })
}

fn allowed(module: &str, name: &str) -> bool {
    matches!((module, name), ("collections", "OrderedDict") | ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_parameter"))
        || (module == "torch" && storage_dtype(name).is_some())
}

fn int(v: &V) -> R<i64> {
    match v {
        V::Int(i) => Ok(*i),
        V::Bool(b) => Ok(*b as i64),
        _ => Err(format!("pickle: expected an integer, got {v:?}").into()),
    }
}

fn usizes(v: &V) -> R<Vec<usize>> {
    match v {
        V::Tuple(t) | V::List(t) => t.iter().map(|x| int(x).and_then(|i| usize::try_from(i).map_err(|_| "pickle: negative size".into()))).collect(),
        _ => Err("pickle: expected a tuple of sizes".into()),
    }
}

fn call(f: V, args: V) -> R<V> {
    let V::Global(m, n) = f else { return Err("pickle: REDUCE on a non-callable".into()) };
    let V::Tuple(a) = args else { return Err("pickle: REDUCE arguments are not a tuple".into()) };
    match (m.as_str(), n.as_str()) {
        ("collections", "OrderedDict") if a.is_empty() => Ok(V::Dict(vec![])),
        ("torch._utils", "_rebuild_tensor_v2") if a.len() >= 4 => {
            let V::Storage(dtype, key) = &a[0] else { return Err("pickle: tensor without a storage".into()) };
            let offset = usize::try_from(int(&a[1])?).map_err(|_| "pickle: negative offset")?;
            let (shape, stride) = (usizes(&a[2])?, usizes(&a[3])?);
            if shape.len() != stride.len() {
                return Err("pickle: tensor shape and stride differ in rank".into());
            }
            Ok(V::Tensor(TensorRef { dtype: *dtype, key: key.clone(), offset, shape, stride }))
        }
        ("torch._utils", "_rebuild_parameter") if !a.is_empty() => Ok(a.into_iter().next().unwrap_or(V::None)),
        _ => Err(format!("pickle: call of {m}.{n} with these arguments is not supported").into()),
    }
}

/// Runs the pickle with the whitelist above; returns the top-level object.
fn unpickle(p: &[u8]) -> R<V> {
    let mut st: Vec<V> = vec![];
    let mut memo: HashMap<u32, V> = HashMap::new();
    let mut copied = 0usize;
    let mut copy = |v: &V| -> R<V> {
        copied += v.size(MAX_COPIED);
        if copied > MAX_COPIED {
            return Err("pickle: too many copied values".into());
        }
        Ok(v.clone())
    };
    let nest = |v: V| -> R<V> {
        if v.deeper_than(MAX_DEPTH) {
            return Err("pickle: nested too deeply".into());
        }
        Ok(v)
    };
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> R<&[u8]> {
        let s = p.get(*i..i.checked_add(n).ok_or("pickle: overflow")?).ok_or("pickle: truncated")?;
        *i += n;
        Ok(s)
    };
    let pop = |st: &mut Vec<V>| st.pop().ok_or_else(|| "pickle: stack underflow".into());
    let pop_mark = |st: &mut Vec<V>| -> R<Vec<V>> {
        let m = st.iter().rposition(|v| matches!(v, V::Mark)).ok_or("pickle: no MARK")?;
        let items = st.split_off(m + 1);
        st.pop();
        Ok(items)
    };
    let set_items = |d: &mut V, items: Vec<V>| -> R<()> {
        let V::Dict(d) = d else { return Err("pickle: SETITEM on a non-dict".into()) };
        if !items.len().is_multiple_of(2) {
            return Err("pickle: SETITEMS with an odd number of items".into());
        }
        let mut it = items.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            if k.deeper_than(MAX_DEPTH) || v.deeper_than(MAX_DEPTH) {
                return Err("pickle: nested too deeply".into());
            }
            d.push((k, v));
        }
        Ok(())
    };
    loop {
        if st.len() > MAX_STACK {
            return Err("pickle: stack too deep".into());
        }
        let op = *take(&mut i, 1)?.first().unwrap();
        match op {
            0x80 => {
                take(&mut i, 1)?; // PROTO
            }
            0x95 => {
                take(&mut i, 8)?; // FRAME
            }
            b'.' => return pop(&mut st),
            b'}' => st.push(V::Dict(vec![])),
            b']' => st.push(V::List(vec![])),
            b')' => st.push(V::Tuple(vec![])),
            b'(' => st.push(V::Mark),
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'K' => st.push(V::Int(take(&mut i, 1)?[0] as i64)),
            b'M' => st.push(V::Int(u16::from_le_bytes(take(&mut i, 2)?.try_into()?) as i64)),
            b'J' => st.push(V::Int(i32::from_le_bytes(take(&mut i, 4)?.try_into()?) as i64)),
            0x8a => {
                // LONG1: little-endian two's complement of n bytes
                let n = take(&mut i, 1)?[0] as usize;
                let b = take(&mut i, n)?;
                if n > 8 {
                    return Err("pickle: integer too large".into());
                }
                let mut v: i64 = 0;
                for (k, &x) in b.iter().enumerate() {
                    v |= (x as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                st.push(V::Int(v));
            }
            b'G' => {
                take(&mut i, 8)?;
                st.push(V::Float);
            }
            b'X' => {
                let n = u32::from_le_bytes(take(&mut i, 4)?.try_into()?) as usize;
                st.push(V::Str(std::str::from_utf8(take(&mut i, n)?)?.to_string()));
            }
            0x8c => {
                let n = take(&mut i, 1)?[0] as usize;
                st.push(V::Str(std::str::from_utf8(take(&mut i, n)?)?.to_string()));
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                memo.insert(k, copy(st.last().ok_or("pickle: BINPUT on an empty stack")?)?);
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into()?);
                memo.insert(k, copy(st.last().ok_or("pickle: LONG_BINPUT on an empty stack")?)?);
            }
            0x94 => {
                let k = memo.len() as u32;
                memo.insert(k, copy(st.last().ok_or("pickle: MEMOIZE on an empty stack")?)?);
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                st.push(copy(memo.get(&k).ok_or("pickle: BINGET of an unknown key")?)?);
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into()?);
                st.push(copy(memo.get(&k).ok_or("pickle: LONG_BINGET of an unknown key")?)?);
            }
            b't' => {
                let items = pop_mark(&mut st)?;
                st.push(nest(V::Tuple(items))?);
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                if st.len() < n {
                    return Err("pickle: stack underflow".into());
                }
                let items = st.split_off(st.len() - n);
                st.push(nest(V::Tuple(items))?);
            }
            b's' => {
                let v = pop(&mut st)?;
                let k = pop(&mut st)?;
                set_items(st.last_mut().ok_or("pickle: SETITEM without a dict")?, vec![k, v])?;
            }
            b'u' => {
                let items = pop_mark(&mut st)?;
                set_items(st.last_mut().ok_or("pickle: SETITEMS without a dict")?, items)?;
            }
            b'a' | b'e' => {
                let items = if op == b'a' { vec![pop(&mut st)?] } else { pop_mark(&mut st)? };
                if items.iter().any(|v| v.deeper_than(MAX_DEPTH)) {
                    return Err("pickle: nested too deeply".into());
                }
                let Some(V::List(l)) = st.last_mut() else { return Err("pickle: APPEND on a non-list".into()) };
                l.extend(items);
            }
            b'c' => {
                // GLOBAL "module\nname\n"
                let mut line = || -> R<String> {
                    let end = p[i..].iter().position(|&c| c == b'\n').ok_or("pickle: truncated GLOBAL")?;
                    let s = std::str::from_utf8(&p[i..i + end])?.to_string();
                    i += end + 1;
                    Ok(s)
                };
                let (m, n) = (line()?, line()?);
                if !allowed(&m, &n) {
                    return Err(format!("pickle: refusing global {m}.{n} (only tensors and dicts are loaded)").into());
                }
                st.push(V::Global(m, n));
            }
            0x93 => {
                let n = pop(&mut st)?;
                let m = pop(&mut st)?;
                let (V::Str(m), V::Str(n)) = (m, n) else { return Err("pickle: STACK_GLOBAL needs strings".into()) };
                if !allowed(&m, &n) {
                    return Err(format!("pickle: refusing global {m}.{n} (only tensors and dicts are loaded)").into());
                }
                st.push(V::Global(m, n));
            }
            b'R' => {
                let args = pop(&mut st)?;
                let f = pop(&mut st)?;
                st.push(call(f, args)?);
            }
            b'Q' => {
                // BINPERSID: ('storage', <torch.XStorage>, key, location, numel)
                let pid = pop(&mut st)?;
                let V::Tuple(t) = pid else { return Err("pickle: persistent id is not a tuple".into()) };
                match (t.first(), t.get(1), t.get(2)) {
                    (Some(V::Str(kind)), Some(V::Global(m, n)), Some(V::Str(key))) if kind == "storage" && m == "torch" => {
                        let dt = storage_dtype(n).ok_or_else(|| format!("pickle: unsupported storage {n}"))?;
                        st.push(V::Storage(dt, key.clone()));
                    }
                    _ => return Err("pickle: unsupported persistent id".into()),
                }
            }
            b'b' => {
                // BUILD: only the attribute dict an OrderedDict state_dict carries (`_metadata`) is accepted, and dropped
                let state = pop(&mut st)?;
                match (st.last(), &state) {
                    (Some(V::Dict(_)), V::Dict(_) | V::None) => {}
                    _ => return Err("pickle: BUILD on this object is not supported".into()),
                }
            }
            0x30 => {
                pop(&mut st)?; // POP
            }
            b'2' => {
                let v = copy(st.last().ok_or("pickle: DUP on an empty stack")?)?;
                st.push(v);
            }
            _ => return Err(format!("pickle: opcode 0x{op:02x} at byte {} is not supported", i - 1).into()),
        }
    }
}

/// Entries of a zip archive: name -> (data offset, size). Stored (uncompressed) entries only.
fn zip_index(b: &[u8]) -> R<HashMap<String, (usize, usize)>> {
    let at = |i: usize, n: usize| -> R<&[u8]> { i.checked_add(n).and_then(|e| b.get(i..e)).ok_or_else(|| "zip: truncated".into()) };
    let u16le = |i: usize| -> R<usize> { Ok(u16::from_le_bytes(at(i, 2)?.try_into()?) as usize) };
    let u32le = |i: usize| -> R<usize> { Ok(u32::from_le_bytes(at(i, 4)?.try_into()?) as usize) };
    let u64le = |i: usize| -> R<usize> { Ok(usize::try_from(u64::from_le_bytes(at(i, 8)?.try_into()?))?) };
    // end of central directory: the last "PK\5\6" within the final 64 KiB + 22 bytes
    let lo = b.len().saturating_sub(65_557);
    let eocd = (lo..b.len().saturating_sub(21)).rev().find(|&i| b[i..i + 4] == [0x50, 0x4b, 5, 6]).ok_or("zip: no end of central directory")?;
    let (mut n, mut cd) = (u16le(eocd + 10)?, u32le(eocd + 16)?);
    if (n == 0xffff || cd == 0xffff_ffff) && eocd >= 20 && b[eocd - 20..eocd - 16] == [0x50, 0x4b, 6, 7] {
        let z = u64le(eocd - 12)?; // zip64 end of central directory record
        if at(z, 4)? != [0x50, 0x4b, 6, 6] {
            return Err("zip: bad zip64 record".into());
        }
        n = u64le(z + 32)?;
        cd = u64le(z + 48)?;
    }
    let mut out = HashMap::new();
    let mut p = cd;
    for _ in 0..n {
        if at(p, 4)? != [0x50, 0x4b, 1, 2] {
            return Err("zip: bad central directory entry".into());
        }
        let method = u16le(p + 10)?;
        let (mut csize, mut usize_, mut local) = (u32le(p + 20)?, u32le(p + 24)?, u32le(p + 42)?);
        let (nl, xl, cl) = (u16le(p + 28)?, u16le(p + 30)?, u16le(p + 32)?);
        let name = std::str::from_utf8(at(p + 46, nl)?)?.to_string();
        // zip64 extra field: the 0xffffffff fields follow in order (uncompressed, compressed, local offset)
        let mut x = p + 46 + nl;
        let xend = x + xl;
        while x + 4 <= xend {
            let (id, len) = (u16le(x)?, u16le(x + 2)?);
            if id == 1 {
                let mut q = x + 4;
                for f in [&mut usize_, &mut csize, &mut local] {
                    if *f == 0xffff_ffff && q + 8 <= x + 4 + len {
                        *f = u64le(q)?;
                        q += 8;
                    }
                }
            }
            x += 4 + len;
        }
        if method != 0 || csize != usize_ {
            return Err(format!("zip: {name} is compressed (only stored entries are supported)").into());
        }
        if at(local, 4)? != [0x50, 0x4b, 3, 4] {
            return Err(format!("zip: bad local header for {name}").into());
        }
        let data = u16le(local + 26)? + u16le(local + 28)? + local + 30;
        if data.checked_add(csize).is_none_or(|e| e > b.len()) {
            return Err(format!("zip: {name} runs past the end of the file").into());
        }
        out.insert(name, (data, csize));
        p += 46 + nl + xl + cl;
    }
    Ok(out)
}

/// A loaded checkpoint: the tensors of its `state_dict` (or of the top-level dict when there is none).
pub struct Pth {
    bytes: Vec<u8>,
    entries: HashMap<String, (usize, usize)>,
    root: String,
    pub tensors: HashMap<String, TensorRef>,
}

impl Pth {
    pub fn open(path: &str) -> R<Self> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::from_bytes(bytes).map_err(|e| format!("{path}: {e}").into())
    }

    pub fn from_bytes(bytes: Vec<u8>) -> R<Self> {
        let entries = zip_index(&bytes)?;
        let pkl = entries.keys().find(|k| k.ends_with("/data.pkl") && k.matches('/').count() == 1).ok_or("not a torch.save archive (no data.pkl)")?;
        let root = pkl.trim_end_matches("data.pkl").to_string();
        if let Some(&(o, n)) = entries.get(&format!("{root}byteorder"))
            && &bytes[o..o + n] != b"little"
        {
            return Err("big-endian checkpoints are not supported".into());
        }
        let (o, n) = entries[pkl];
        let top = unpickle(&bytes[o..o + n])?;
        let V::Dict(items) = top else { return Err("checkpoint is not a dict".into()) };
        let sd = items.iter().find(|(k, _)| matches!(k, V::Str(s) if s == "state_dict")).map(|(_, v)| v.clone());
        let items = match sd {
            Some(V::Dict(d)) => d,
            Some(_) => return Err("state_dict is not a dict".into()),
            None => items,
        };
        let mut tensors = HashMap::new();
        for (k, v) in items {
            if let (V::Str(k), V::Tensor(t)) = (k, v) {
                tensors.insert(k, t);
            }
        }
        Ok(Pth { bytes, entries, root, tensors })
    }

    pub fn get(&self, name: &str) -> R<&TensorRef> {
        self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}").into())
    }

    /// A tensor as f32 in row-major order (any strides; float storages only).
    pub fn f32s(&self, name: &str) -> R<(Vec<f32>, Vec<usize>)> {
        let t = self.get(name)?;
        let &(o, n) = self.entries.get(&format!("{}data/{}", self.root, t.key)).ok_or_else(|| format!("{name}: storage {} missing", t.key))?;
        let data = &self.bytes[o..o + n];
        let es = t.dtype.size();
        let numel = t.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| format!("{name}: too many elements"))?;
        if numel == 0 {
            return Ok((vec![], t.shape.clone()));
        }
        // strides of 0 let a few stored bytes describe a huge tensor; no AudioVAE weight comes near this
        if numel > MAX_ELEMENTS {
            return Err(format!("{name}: {numel} elements is more than this reader accepts").into());
        }
        // the largest element index addressed must lie inside the storage
        let end = t
            .shape
            .iter()
            .zip(&t.stride)
            .try_fold(t.offset, |e, (&d, &s)| (d - 1).checked_mul(s).and_then(|x| e.checked_add(x)))
            .and_then(|last| last.checked_add(1)?.checked_mul(es));
        if end.is_none_or(|e| e > data.len()) {
            return Err(format!("{name}: tensor runs past its storage").into());
        }
        let read = |e: usize| -> f32 {
            let b = &data[e * es..e * es + es];
            match t.dtype {
                DType::F32 => f32::from_le_bytes(b.try_into().unwrap()),
                DType::F16 => f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())),
                DType::BF16 => f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16),
                DType::I32 => i32::from_le_bytes(b.try_into().unwrap()) as f32,
                DType::I64 => i64::from_le_bytes(b.try_into().unwrap()) as f32,
            }
        };
        // row-major tensors (the usual case) are one contiguous run of the storage
        let mut contiguous = true;
        let mut expect = 1;
        for (&d, &st) in t.shape.iter().zip(&t.stride).rev() {
            contiguous &= d == 1 || st == expect;
            expect *= d;
        }
        if contiguous {
            let run = &data[t.offset * es..(t.offset + numel) * es];
            let v = match t.dtype {
                DType::F32 => run.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
                _ => (t.offset..t.offset + numel).map(read).collect(),
            };
            return Ok((v, t.shape.clone()));
        }
        let mut out = Vec::with_capacity(numel);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..numel {
            out.push(read(t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>()));
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < t.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok((out, t.shape.clone()))
    }

    /// `torch.nn.utils.weight_norm` folded: weight = g * v / ||v||, the norm over every dim but the first.
    pub fn weight_norm(&self, prefix: &str) -> R<(Vec<f32>, Vec<usize>)> {
        let (g, gs) = self.f32s(&format!("{prefix}.weight_g"))?;
        let (v, vs) = self.f32s(&format!("{prefix}.weight_v"))?;
        let rows = *vs.first().ok_or("weight_v is a scalar")?;
        if v.is_empty() {
            return Err(format!("{prefix}: weight_v is empty").into());
        }
        if gs.first() != Some(&rows) || g.len() != rows {
            return Err(format!("{prefix}: weight_g does not match weight_v").into());
        }
        let per = v.len() / rows;
        let mut w = vec![0f32; v.len()];
        let fold = |r0: usize, out: &mut [f32]| {
            for (i, o) in out.chunks_mut(per).enumerate() {
                let row = &v[(r0 + i) * per..(r0 + i + 1) * per];
                let norm = row.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() as f32;
                let s = g[r0 + i] / norm;
                for (o, &x) in o.iter_mut().zip(row) {
                    *o = x * s;
                }
            }
        };
        let threads = if v.len() < 1 << 18 { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()) };
        let rows_per = rows.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (i, out) in w.chunks_mut(rows_per * per.max(1)).enumerate() {
                let fold = &fold;
                sc.spawn(move || fold(i * rows_per, out));
            }
        });
        Ok((w, vs))
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (31, _) => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored zip with the given entries.
    fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let (mut out, mut cd) = (vec![], vec![]);
        for (name, data) in entries {
            let off = out.len() as u32;
            out.extend_from_slice(&[0x50, 0x4b, 3, 4, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            cd.extend_from_slice(&[0x50, 0x4b, 1, 2, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(data.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
            cd.extend_from_slice(&[0; 12]);
            cd.extend_from_slice(&off.to_le_bytes());
            cd.extend_from_slice(name.as_bytes());
        }
        let cd_off = out.len() as u32;
        out.extend_from_slice(&cd);
        out.extend_from_slice(&[0x50, 0x4b, 5, 6, 0, 0, 0, 0]);
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(cd.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    /// pickle of {'state_dict': OrderedDict(w=<2x2 float tensor on storage "0" with the given stride>)}
    fn pickle_one_tensor(stride: &[u8]) -> Vec<u8> {
        let mut p = vec![0x80, 2, b'}', b'q', 0, b'X'];
        p.extend_from_slice(&10u32.to_le_bytes());
        p.extend_from_slice(b"state_dict");
        p.extend_from_slice(b"ccollections\nOrderedDict\nq\x01)Rq\x02X");
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(b"w");
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\nq\x03((X");
        p.extend_from_slice(&7u32.to_le_bytes());
        p.extend_from_slice(b"storagectorch\nFloatStorage\nX");
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(b"0X");
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(b"cpuK\x04tQK\x00K\x02K\x02\x86");
        p.extend_from_slice(stride);
        // requires_grad False, backward hooks OrderedDict(); call; w = tensor; state_dict = OrderedDict
        p.extend_from_slice(b"\x89ccollections\nOrderedDict\n)RtRss.");
        p
    }

    #[test]
    fn reads_tensor_and_strides() {
        let data: Vec<u8> = [1f32, 2., 3., 4.].iter().flat_map(|x| x.to_le_bytes()).collect();
        for (stride, want) in [(&b"K\x02K\x01\x86"[..], [1f32, 2., 3., 4.]), (&b"K\x01K\x02\x86"[..], [1., 3., 2., 4.])] {
            let pkl = pickle_one_tensor(stride);
            let z = zip(&[("m/data.pkl", &pkl), ("m/data/0", &data), ("m/byteorder", b"little")]);
            let p = Pth::from_bytes(z).unwrap();
            let (v, shape) = p.f32s("w").unwrap();
            assert_eq!(shape, vec![2, 2]);
            assert_eq!(v, want);
        }
    }

    #[test]
    fn refuses_other_globals() {
        let pkl = b"\x80\x02cos\nsystem\nX\x02\x00\x00\x00ls\x85R.".to_vec();
        let z = zip(&[("m/data.pkl", &pkl)]);
        let e = Pth::from_bytes(z).err().unwrap().to_string();
        assert!(e.contains("refusing global os.system"), "{e}");
        let pkl = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93.".to_vec();
        let e = Pth::from_bytes(zip(&[("m/data.pkl", &pkl)])).err().unwrap().to_string();
        assert!(e.contains("refusing global builtins.eval"), "{e}");
    }

    #[test]
    fn rejects_out_of_range_tensor_and_garbage() {
        let data: Vec<u8> = [1f32, 2., 3.].iter().flat_map(|x| x.to_le_bytes()).collect(); // one element short
        let pkl = pickle_one_tensor(b"K\x02K\x01\x86");
        let p = Pth::from_bytes(zip(&[("m/data.pkl", &pkl), ("m/data/0", &data)])).unwrap();
        assert!(p.f32s("w").is_err());
        for n in 0..pkl.len() {
            let _ = Pth::from_bytes(zip(&[("m/data.pkl", &pkl[..n]), ("m/data/0", &data)])); // truncated pickles: errors, no panic
        }
        assert!(Pth::from_bytes(b"PK not a zip".to_vec()).is_err());
        assert!(Pth::from_bytes(vec![]).is_err());
    }

    #[test]
    fn bounds_copies_and_nesting() {
        // a list that DUPs itself into a tuple pair 40 times would be 2^40 values
        let mut pkl = b"\x80\x02]".to_vec();
        pkl.extend(std::iter::repeat_n(&b"2\x86"[..], 40).flatten());
        pkl.push(b'.');
        let e = Pth::from_bytes(zip(&[("m/data.pkl", &pkl)])).err().unwrap().to_string();
        assert!(e.contains("too many copied values") || e.contains("nested too deeply"), "{e}");
        let mut pkl = b"\x80\x02N".to_vec();
        pkl.extend(std::iter::repeat_n(0x85u8, 100));
        pkl.push(b'.');
        let e = Pth::from_bytes(zip(&[("m/data.pkl", &pkl)])).err().unwrap().to_string();
        assert!(e.contains("nested too deeply"), "{e}");
    }

    #[test]
    fn weight_norm_folds() {
        // g = [2], v = [[3, 4]] -> w = 2 * [3, 4] / 5
        let mut p = Pth { bytes: vec![], entries: HashMap::new(), root: "m/".into(), tensors: HashMap::new() };
        let mut bytes = vec![];
        for (k, vals, shape) in [("x.weight_g", vec![2f32], vec![1, 1, 1]), ("x.weight_v", vec![3f32, 4.], vec![1, 1, 2])] {
            let o = bytes.len();
            bytes.extend(vals.iter().flat_map(|x| x.to_le_bytes()));
            p.entries.insert(format!("m/data/{k}"), (o, vals.len() * 4));
            let stride = if shape[2] == 2 { vec![2, 2, 1] } else { vec![1, 1, 1] };
            p.tensors.insert(k.into(), TensorRef { dtype: DType::F32, key: k.into(), offset: 0, shape, stride });
        }
        p.bytes = bytes;
        let (w, s) = p.weight_norm("x").unwrap();
        assert_eq!(s, vec![1, 1, 2]);
        assert!((w[0] - 1.2).abs() < 1e-6 && (w[1] - 1.6).abs() < 1e-6);
    }
}
