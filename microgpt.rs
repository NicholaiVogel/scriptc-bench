//! A dependency-free Rust translation of `microgpt.py`.
//!
//! This intentionally keeps the scalar autograd implementation and model
//! structure of the original rather than using a tensor library.

use std::env;
use std::fs;
use std::f64::consts::PI;

// Python-compatible MT19937 bits used by random.seed(42), random.shuffle,
// random.gauss, and random.choices.  Keeping this here makes the port
// deterministic without an external crate.
struct Random {
    mt: [u32; 624],
    index: usize,
    gauss_next: Option<f64>,
}

impl Random {
    fn new(seed: u32) -> Self {
        let mut r = Self { mt: [0; 624], index: 624, gauss_next: None };
        r.mt[0] = 19650218;
        for i in 1..624 { r.mt[i] = 1812433253u32.wrapping_mul(r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_add(i as u32); }
        let key = [seed];
        let mut i = 1usize; let mut j = 0usize; let mut k = 624usize.max(key.len());
        while k > 0 {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1; j += 1;
            if i >= 624 { r.mt[0] = r.mt[623]; i = 1; }
            if j >= key.len() { j = 0; }
            k -= 1;
        }
        k = 623;
        while k > 0 {
            r.mt[i] = (r.mt[i] ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 { r.mt[0] = r.mt[623]; i = 1; }
            k -= 1;
        }
        r.mt[0] = 0x80000000;
        r
    }
    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7fffffff);
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
        }
        self.index = 0;
    }
    fn u32(&mut self) -> u32 {
        if self.index >= 624 { self.twist(); }
        let mut y = self.mt[self.index]; self.index += 1;
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >> 18; y
    }
    fn random(&mut self) -> f64 { (((self.u32() >> 5) as u64 * 67108864 + (self.u32() >> 6) as u64) as f64) / 9007199254740992.0 }
    fn bits(&mut self, k: u32) -> u32 { self.u32() >> (32 - k) }
    fn below(&mut self, n: usize) -> usize {
        let k = (usize::BITS - n.leading_zeros()) as u32;
        loop { let x = self.bits(k) as usize; if x < n { return x; } }
    }
    fn shuffle<T>(&mut self, a: &mut [T]) { for i in (1..a.len()).rev() { let j = self.below(i + 1); a.swap(i, j); } }
    fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        if let Some(x) = self.gauss_next.take() { return mean + std * x; }
        let x2pi = self.random() * (2.0 * PI);
        let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
        self.gauss_next = Some(x2pi.sin() * g2rad);
        mean + std * (x2pi.cos() * g2rad)
    }
    fn choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum(); let target = self.random() * total; let mut cursor = 0.0;
        for (i, w) in weights.iter().enumerate() { cursor += w; if target < cursor { return i; } }
        weights.len() - 1
    }
}

#[derive(Clone)]
struct Value { data: f64, grad: f64, children: Vec<(usize, f64)> }
struct Graph { values: Vec<Value> }
impl Graph {
    fn new() -> Self { Self { values: Vec::new() } }
    fn node(&mut self, data: f64, children: Vec<(usize, f64)>) -> usize { let i = self.values.len(); self.values.push(Value { data, grad: 0.0, children }); i }
    fn c(&mut self, x: f64) -> usize { self.node(x, vec![]) }
    fn add(&mut self, a: usize, b: usize) -> usize { self.node(self.values[a].data + self.values[b].data, vec![(a, 1.0), (b, 1.0)]) }
    fn mul(&mut self, a: usize, b: usize) -> usize { self.node(self.values[a].data * self.values[b].data, vec![(a, self.values[b].data), (b, self.values[a].data)]) }
    fn mulf(&mut self, a: usize, x: f64) -> usize { let b = self.c(x); self.mul(a, b) }
    fn neg(&mut self, a: usize) -> usize { self.mulf(a, -1.0) }
    fn pow(&mut self, a: usize, p: f64) -> usize { self.node(self.values[a].data.powf(p), vec![(a, p * self.values[a].data.powf(p - 1.0))]) }
    fn log(&mut self, a: usize) -> usize { self.node(self.values[a].data.ln(), vec![(a, 1.0 / self.values[a].data)]) }
    fn exp(&mut self, a: usize) -> usize { let x = self.values[a].data.exp(); self.node(x, vec![(a, x)]) }
    fn relu(&mut self, a: usize) -> usize { self.node(self.values[a].data.max(0.0), vec![(a, if self.values[a].data > 0.0 { 1.0 } else { 0.0 })]) }
    fn sum(&mut self, xs: &[usize]) -> usize { xs.iter().fold(self.c(0.0), |s, &x| self.add(s, x)) }
    fn backward(&mut self, root: usize) {
        let mut topo = Vec::new(); let mut seen = vec![false; self.values.len()]; let mut stack = vec![(root, false)];
        while let Some((v, expanded)) = stack.pop() {
            if expanded { topo.push(v); continue; }
            if seen[v] { continue; } seen[v] = true; stack.push((v, true));
            for &(child, _) in self.values[v].children.iter().rev() { stack.push((child, false)); }
        }
        self.values[root].grad = 1.0;
        for &v in topo.iter().rev() { let g = self.values[v].grad; let children = self.values[v].children.clone(); for (child, local) in children { self.values[child].grad += local * g; } }
    }
}

type Matrix = Vec<Vec<usize>>;
fn linear(g: &mut Graph, x: &[usize], w: &Matrix) -> Vec<usize> { let mut out=Vec::new(); for row in w { let mut terms=Vec::new(); for (&a,&b) in row.iter().zip(x) { terms.push(g.mul(a,b)); } out.push(g.sum(&terms)); } out }
fn softmax(g: &mut Graph, logits: &[usize]) -> Vec<usize> {
    let max = logits.iter().map(|&x| g.values[x].data).fold(f64::NEG_INFINITY, f64::max);
    let mut exps=Vec::new(); for &x in logits { let c=g.c(-max); let shifted=g.add(x,c); exps.push(g.exp(shifted)); } let total = g.sum(&exps); let inv=g.pow(total,-1.0);
    exps.iter().map(|&x| g.mul(x, inv)).collect()
}
fn rmsnorm(g: &mut Graph, x: &[usize]) -> Vec<usize> {
    let squares: Vec<_> = x.iter().map(|&a| g.mul(a, a)).collect(); let sum=g.sum(&squares); let ms = g.mulf(sum, 1.0 / x.len() as f64);
    let c=g.c(1e-5); let shifted=g.add(ms,c); let scale = g.pow(shifted, -0.5); x.iter().map(|&a| g.mul(a, scale)).collect()
}

fn gpt(g: &mut Graph, token: usize, pos: usize, keys: &mut [Vec<Vec<usize>>], values: &mut [Vec<Vec<usize>>], state: &[Matrix], _vocab: usize) -> Vec<usize> {
    let x0: Vec<_> = state[0][token].iter().zip(&state[1][pos]).map(|(&a, &b)| g.add(a, b)).collect(); let mut x = rmsnorm(g, &x0);
    let n_head = 4; let head_dim = 4;
    for li in 0..1 {
        let residual = x.clone(); let norm = rmsnorm(g, &x);
        let q = linear(g, &norm, &state[3 + li * 6]); let k = linear(g, &norm, &state[4 + li * 6]); let v = linear(g, &norm, &state[5 + li * 6]);
        keys[li].push(k); values[li].push(v); let mut attn = Vec::new();
        for h in 0..n_head { let hs = h * head_dim; let mut scores = Vec::new();
            for t in 0..keys[li].len() { let dots: Vec<_> = (0..head_dim).map(|j| g.mul(q[hs+j], keys[li][t][hs+j])).collect(); let dot_sum=g.sum(&dots); scores.push(g.mulf(dot_sum, 1.0 / (head_dim as f64).sqrt())); }
            let weights = softmax(g, &scores); for j in 0..head_dim { let terms: Vec<_> = weights.iter().enumerate().map(|(t, &a)| g.mul(a, values[li][t][hs+j])).collect(); attn.push(g.sum(&terms)); }
        }
        let mut y = linear(g, &attn, &state[6 + li * 6]); y = y.iter().zip(&residual).map(|(&a, &b)| g.add(a, b)).collect();
        let residual = y.clone(); let z = rmsnorm(g, &y); let z = linear(g, &z, &state[7 + li * 6]); let z = z.iter().map(|&a| g.relu(a)).collect::<Vec<_>>();
        let z = linear(g, &z, &state[8 + li * 6]); x = z.iter().zip(&residual).map(|(&a, &b)| g.add(a, b)).collect();
    }
    linear(g, &x, &state[2 + 0])
}

fn main() {
    let mut rng = Random::new(42); let input = fs::read_to_string("input.txt").expect("input.txt"); let mut docs: Vec<String> = input.lines().map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect(); rng.shuffle(&mut docs); println!("num docs: {}", docs.len());
    let mut uchars: Vec<char> = docs.iter().flat_map(|s| s.chars()).collect(); uchars.sort_unstable(); uchars.dedup(); let bos = uchars.len(); let vocab = bos + 1; println!("vocab size: {vocab}");
    let mut graph = Graph::new(); let matrix = |g: &mut Graph, nout: usize, nin: usize, rng: &mut Random| -> Matrix { (0..nout).map(|_| (0..nin).map(|_| g.c(rng.gauss(0.0, 0.08))).collect()).collect() };
    // wte, wpe, lm_head, then each layer's six matrices.
    let mut state = vec![matrix(&mut graph, vocab, 16, &mut rng), matrix(&mut graph, 16, 16, &mut rng), matrix(&mut graph, vocab, 16, &mut rng)];
    state.extend([matrix(&mut graph,16,16,&mut rng), matrix(&mut graph,16,16,&mut rng), matrix(&mut graph,16,16,&mut rng), matrix(&mut graph,16,16,&mut rng), matrix(&mut graph,64,16,&mut rng), matrix(&mut graph,16,64,&mut rng)]);
    let params: Vec<_> = state.iter().flat_map(|m| m.iter().flat_map(|r| r.iter().copied())).collect(); println!("num params: {}", params.len());
    let steps: usize = env::var("MICROGPT_STEPS").ok().and_then(|x| x.parse().ok()).unwrap_or(1000); let (mut m, mut v) = (vec![0.0; params.len()], vec![0.0; params.len()]);
    for step in 0..steps { let chars: Vec<char> = docs[step % docs.len()].chars().collect(); let mut tokens = vec![bos]; tokens.extend(chars.iter().map(|c| uchars.binary_search(c).unwrap())); tokens.push(bos); let n = 16.min(tokens.len() - 1); let mut keys = vec![Vec::new()]; let mut values = vec![Vec::new()]; let mut losses = Vec::new();
        for pos in 0..n { let logits = gpt(&mut graph, tokens[pos], pos, &mut keys, &mut values, &state, vocab); let probs = softmax(&mut graph, &logits); let logp=graph.log(probs[tokens[pos+1]]); losses.push(graph.neg(logp)); }
        let loss_sum=graph.sum(&losses); let loss = graph.mulf(loss_sum, 1.0 / n as f64); graph.backward(loss); let lr = 0.01 * (1.0 - step as f64 / steps as f64);
        for (i, &p) in params.iter().enumerate() { let grad = graph.values[p].grad; m[i]=0.85*m[i]+0.15*grad; v[i]=0.99*v[i]+0.01*grad*grad; let mh=m[i]/(1.0-0.85_f64.powi((step+1) as i32)); let vh=v[i]/(1.0-0.99_f64.powi((step+1) as i32)); graph.values[p].data -= lr*mh/(vh.sqrt()+1e-8); graph.values[p].grad=0.0; }
        println!("step {:4} / {:4} | loss {:.4}", step+1, steps, graph.values[loss].data);
    }
    println!("--- inference (new, hallucinated names) ---"); for sample_idx in 0..20 { let mut keys=vec![Vec::new()]; let mut values=vec![Vec::new()]; let mut token=bos; let mut sample=String::new(); for pos in 0..16 { let logits=gpt(&mut graph,token,pos,&mut keys,&mut values,&state,vocab); let scaled: Vec<_>=logits.iter().map(|&x|graph.mulf(x,2.0)).collect(); let probs=softmax(&mut graph,&scaled); let weights:Vec<_>=probs.iter().map(|&x|graph.values[x].data).collect(); token=rng.choice(&weights); if token==bos {break} sample.push(uchars[token]); } println!("sample {:2}: {}",sample_idx+1,sample); }
}
