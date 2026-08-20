/**
 * microGPT with a flat scalar tape rather than one object per autograd node.
 *
 * Parameters and graph nodes are integer ids into parallel arrays. Because
 * every operation is appended after its inputs, reverse node order is already
 * a valid backward traversal. This keeps the reference algorithm intact while
 * removing Value objects, child arrays, local-gradient arrays, and Set/DFS
 * bookkeeping from the hot path.
 */
import { readFileSync } from "node:fs";

class Rng {
	state: number;
	spare: number | null = null;
	constructor(seed: number) { this.state = seed >>> 0; }
	next(): number {
		let x = this.state; x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
		this.state = x >>> 0; return this.state / 4294967296;
	}
	gauss(mean: number, std: number): number {
		if (this.spare !== null) { const v = this.spare; this.spare = null; return mean + std * v; }
		const r = Math.sqrt(-2 * Math.log(Math.max(this.next(), 1e-12)));
		const t = 2 * Math.PI * this.next(); this.spare = r * Math.sin(t);
		return mean + std * r * Math.cos(t);
	}
	shuffle(items: string[]): void {
		for (let i = items.length - 1; i > 0; i -= 1) { const j = Math.floor(this.next() * (i + 1)); const v = items[i]; items[i] = items[j]; items[j] = v; }
	}
	choices(weights: number[]): number {
		let total = 0; for (const w of weights) total += w;
		const target = this.next() * total; let cursor = 0;
		for (let i = 0; i < weights.length; i += 1) { cursor += weights[i]; if (target < cursor) return i; }
		return weights.length - 1;
	}
}

// Flat tape: ids are indexes; parameter ids occupy the prefix of the arrays.
const data: number[] = [], grad: number[] = [], left: number[] = [], right: number[] = [], leftGrad: number[] = [], rightGrad: number[] = [], kind: number[] = [], fanStart: number[] = [], fanCount: number[] = [], fanA: number[] = [], fanB: number[] = [];
let nodeCount = 0;
let fanUsed = 0;
function node(value: number, a = -1, b = -1, ga = 0, gb = 0): number {
	const id = nodeCount; nodeCount += 1;
	data[id] = value; grad[id] = 0; left[id] = a; right[id] = b; leftGrad[id] = ga; rightGrad[id] = gb; kind[id] = 0; fanStart[id] = 0; fanCount[id] = 0; return id;
}
function dot(a: number[], b: number[]): number {
	const id = node(0); let value = 0; const start = fanUsed;
	for (let i = 0; i < a.length; i += 1) { value += data[a[i]] * data[b[i]]; fanA[fanUsed] = a[i]; fanB[fanUsed] = b[i]; fanUsed += 1; }
	data[id] = value; kind[id] = 1; fanStart[id] = start; fanCount[id] = a.length; return id;
}
function constant(value: number): number { return node(value); }
function plus(a: number, b: number): number { return node(data[a] + data[b], a, b, 1, 1); }
function times(a: number, b: number): number { return node(data[a] * data[b], a, b, data[b], data[a]); }
function power(a: number, exponent: number): number { return node(data[a] ** exponent, a, -1, exponent * data[a] ** (exponent - 1)); }
function logarithm(a: number): number { return node(Math.log(data[a]), a, -1, 1 / data[a]); }
function exponential(a: number): number { const v = Math.exp(data[a]); return node(v, a, -1, v); }
function relu(a: number): number { return node(Math.max(0, data[a]), a, -1, Number(data[a] > 0)); }
function addScalar(a: number, value: number): number { return node(data[a] + value, a, -1, 1); }
function scale(a: number, value: number): number { return node(data[a] * value, a, -1, value); }
function addAll(values: number[]): number {
	if (values.length === 0) return constant(0);
	let result = values[0]; for (let i = 1; i < values.length; i += 1) result = plus(result, values[i]); return result;
}
function backward(loss: number, parameterCount: number): void {
	grad[loss] = 1;
	for (let i = loss; i >= parameterCount; i -= 1) {
		const g = grad[i]; const a = left[i]; const b = right[i];
		if (kind[i] === 1) { const start = fanStart[i]; for (let j = 0; j < fanCount[i]; j += 1) { const p = fanA[start + j]; const q = fanB[start + j]; grad[p] = grad[p] + data[q] * g; grad[q] = grad[q] + data[p] * g; } }
		else { if (a >= 0) grad[a] = grad[a] + leftGrad[i] * g; if (b >= 0) grad[b] = grad[b] + rightGrad[i] * g; }
	}
}

type Matrix = number[][];
const rng = new Rng(42);
const inputPath = process.argv.length > 2 ? process.argv[2] : "../input.txt";
const docs = readFileSync(inputPath, "utf8").split(/\r?\n/).filter((line) => line.length > 0);
rng.shuffle(docs); console.log(`num docs: ${docs.length}`);
const chars: string[] = [];
for (const char of docs.join("")) if (chars.indexOf(char) < 0) chars.push(char);
chars.sort();
const BOS = chars.length; const vocabSize = chars.length + 1;
console.log(`vocab size: ${vocabSize}`);
const nLayer = 1, nEmbd = 16, blockSize = 16, nHead = 4, headDim = nEmbd / nHead;
function matrix(nout: number, nin: number, std = 0.08): Matrix {
	return Array.from({ length: nout }, () => Array.from({ length: nin }, () => node(rng.gauss(0, std))));
}
const state: Record<string, Matrix> = { wte: matrix(vocabSize, nEmbd), wpe: matrix(blockSize, nEmbd), lmHead: matrix(vocabSize, nEmbd) };
for (let i = 0; i < nLayer; i += 1) {
	state[`layer${i}.attnWq`] = matrix(nEmbd, nEmbd); state[`layer${i}.attnWk`] = matrix(nEmbd, nEmbd);
	state[`layer${i}.attnWv`] = matrix(nEmbd, nEmbd); state[`layer${i}.attnWo`] = matrix(nEmbd, nEmbd);
	state[`layer${i}.mlpFc1`] = matrix(4 * nEmbd, nEmbd); state[`layer${i}.mlpFc2`] = matrix(nEmbd, 4 * nEmbd);
}
const params: number[] = [];
for (const m of Object.values(state)) for (const row of m) for (const id of row) params.push(id);
console.log(`num params: ${params.length}`);

function linear(x: number[], weights: Matrix): number[] {
	return weights.map((row) => dot(row, x));
}
function softmax(logits: number[]): number[] {
	let max = -Infinity; for (const id of logits) if (data[id] > max) max = data[id];
	const exps = logits.map((id) => exponential(addScalar(id, -max)));
	const total = addAll(exps); const inverse = power(total, -1);
	return exps.map((id) => times(id, inverse));
}
function rmsnorm(x: number[]): number[] {
	const squares = x.map((id) => times(id, id));
	const meanSquare = scale(addAll(squares), 1 / x.length);
	const scaleId = power(addScalar(meanSquare, 1e-5), -0.5);
	return x.map((id) => times(id, scaleId));
}
function gpt(tokenId: number, posId: number, keys: number[][][], values: number[][][]): number[] {
	let x = state.wte[tokenId].map((token, i) => plus(token, state.wpe[posId][i])); x = rmsnorm(x);
	for (let layer = 0; layer < nLayer; layer += 1) {
		const residual = x; x = rmsnorm(x);
		const q = linear(x, state[`layer${layer}.attnWq`]); const k = linear(x, state[`layer${layer}.attnWk`]); const v = linear(x, state[`layer${layer}.attnWv`]);
		keys[layer].push(k); values[layer].push(v); const attention: number[] = [];
		for (let head = 0; head < nHead; head += 1) {
			const start = head * headDim; const logits: number[] = [];
			for (const past of keys[layer]) { const qa: number[] = [], pa: number[] = []; for (let j = 0; j < headDim; j += 1) { qa.push(q[start + j]); pa.push(past[start + j]); } logits.push(scale(dot(qa, pa), 1 / Math.sqrt(headDim))); }
			const weights = softmax(logits);
			for (let j = 0; j < headDim; j += 1) { const wa: number[] = [], va: number[] = []; for (let t = 0; t < values[layer].length; t += 1) { wa.push(weights[t]); va.push(values[layer][t][start + j]); } attention.push(dot(wa, va)); }
		}
		x = linear(attention, state[`layer${layer}.attnWo`]).map((id, i) => plus(id, residual[i]));
		const mlpResidual = x; x = rmsnorm(x); x = linear(x, state[`layer${layer}.mlpFc1`]).map(relu);
		x = linear(x, state[`layer${layer}.mlpFc2`]).map((id, i) => plus(id, mlpResidual[i]));
	}
	return linear(x, state.lmHead);
}

const moments = Array.from({ length: params.length }, () => 0), velocities = Array.from({ length: params.length }, () => 0);
const numSteps = process.env.MICROGPT_STEPS === undefined ? 1000 : parseInt(process.env.MICROGPT_STEPS, 10);
for (let step = 0; step < numSteps; step += 1) {
	for (const id of params) grad[id] = 0;
	nodeCount = params.length;
	fanUsed = 0;
	const doc = docs[step % docs.length]; const tokens = [BOS, ...doc.split("").map((char) => chars.indexOf(char)), BOS]; const n = Math.min(blockSize, tokens.length - 1);
	const keys = Array.from({ length: nLayer }, () => [] as number[][]), values = Array.from({ length: nLayer }, () => [] as number[][]); const losses: number[] = [];
	for (let pos = 0; pos < n; pos += 1) { const probabilities = softmax(gpt(tokens[pos], pos, keys, values)); losses.push(logarithm(probabilities[tokens[pos + 1]])); }
	const loss = scale(addAll(losses), -1 / n); backward(loss, params.length);
	const lr = 0.01 * (1 - step / numSteps);
	for (let i = 0; i < params.length; i += 1) { const id = params[i]; moments[i] = 0.85 * moments[i] + 0.15 * grad[id]; velocities[i] = 0.99 * velocities[i] + 0.01 * grad[id] ** 2; const mh = moments[i] / (1 - 0.85 ** (step + 1)); const vh = velocities[i] / (1 - 0.99 ** (step + 1)); data[id] = data[id] - lr * mh / (Math.sqrt(vh) + 1e-8); }
	if (step === 0 || (step + 1) % 100 === 0 || step + 1 === numSteps) console.log(`step ${String(step + 1).padStart(4)} / ${String(numSteps).padStart(4)} | loss ${data[loss].toFixed(4)}`);
}
console.log("--- inference (new, hallucinated names) ---");
for (let sampleIndex = 0; sampleIndex < 20; sampleIndex += 1) { const keys = Array.from({ length: nLayer }, () => [] as number[][]), values = Array.from({ length: nLayer }, () => [] as number[][]); let token = BOS; let sample = ""; for (let pos = 0; pos < blockSize; pos += 1) { const logits = gpt(token, pos, keys, values).map((id) => scale(id, 2)); token = rng.choices(softmax(logits).map((id) => data[id])); if (token === BOS) break; sample += chars[token]; } console.log(`sample ${String(sampleIndex + 1).padStart(2)}: ${sample}`); }
