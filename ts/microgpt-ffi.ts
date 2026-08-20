/** Dependency-free TypeScript translation of microgpt.py.
 *
 * The algorithm is intentionally scalar and object-heavy: the point is to
 * compare runtimes on the same small autograd workload, not to optimize GPT.
 * Python's random module is not reproduced bit-for-bit; this port uses a
 * seeded xorshift generator so runs remain deterministic within TypeScript.
 */
import { readFileSync } from "node:fs";

declare function signetSqrt(value: number): number;
declare function signetLog(value: number): number;
declare function signetExp(value: number): number;
declare function signetSin(value: number): number;
declare function signetCos(value: number): number;

function nativeSqrt(value: number): number { return signetSqrt(value); }
function nativeLog(value: number): number { return signetLog(value); }
function nativeExp(value: number): number { return signetExp(value); }
function nativeSin(value: number): number { return signetSin(value); }
function nativeCos(value: number): number { return signetCos(value); }

class Rng {
	private state: number;
	private spare: number | null = null;

	constructor(seed: number) {
		this.state = seed >>> 0;
	}

	next(): number {
		let x = this.state;
		x ^= x << 13;
		x ^= x >>> 17;
		x ^= x << 5;
		this.state = x >>> 0;
		return this.state / 4294967296;
	}

	gauss(mean: number, std: number): number {
		if (this.spare !== null) {
			const value = this.spare;
			this.spare = null;
			return mean + std * value;
		}
		const radius = nativeSqrt(-2 * nativeLog(Math.max(this.next(), 1e-12)));
		const theta = 2 * 3.141592653589793 * this.next();
		this.spare = radius * nativeSin(theta);
		return mean + std * radius * nativeCos(theta);
	}

	shuffle<T>(items: T[]): void {
		for (let i = items.length - 1; i > 0; i -= 1) {
			const j = Math.floor(this.next() * (i + 1));
			const temp = items[i];
			items[i] = items[j];
			items[j] = temp;
		}
	}

	choices(weights: number[]): number {
		const target = this.next() * weights.reduce((sum, weight) => sum + weight, 0);
		let cursor = 0;
		for (let i = 0; i < weights.length; i += 1) {
			cursor += weights[i];
			if (target < cursor) return i;
		}
		return weights.length - 1;
	}
}

let nextValueId = 0;

class Value {
	readonly id: number;
	data: number;
	grad = 0;
	readonly children: Value[];
	readonly localGrads: number[];

	constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
		this.id = nextValueId;
		nextValueId += 1;
		this.data = data;
		this.children = children;
		this.localGrads = localGrads;
	}

	add(other: Value | number): Value {
		const rhs = typeof other === "number" ? new Value(other) : other;
		return new Value(this.data + rhs.data, [this, rhs], [1, 1]);
	}

	mul(other: Value | number): Value {
		const rhs = typeof other === "number" ? new Value(other) : other;
		return new Value(this.data * rhs.data, [this, rhs], [rhs.data, this.data]);
	}

	pow(other: number): Value {
		return new Value(this.data ** other, [this], [other * this.data ** (other - 1)]);
	}

	log(): Value { return new Value(nativeLog(this.data), [this], [1 / this.data]); }
	exp(): Value { const value = nativeExp(this.data); return new Value(value, [this], [value]); }
	relu(): Value { return new Value(Math.max(0, this.data), [this], [Number(this.data > 0)]); }
	neg(): Value { return this.mul(-1); }

	backward(): void {
		const topo: Value[] = [];
		const visited = new Set<number>();
		const stack: Array<{ value: Value; expanded: boolean }> = [{ value: this, expanded: false }];
		while (stack.length > 0) {
			const item = stack.pop();
			if (item === undefined) continue;
			if (item.expanded) {
				topo.push(item.value);
				continue;
			}
			if (visited.has(item.value.id)) continue;
			visited.add(item.value.id);
			stack.push({ value: item.value, expanded: true });
			for (const child of item.value.children) stack.push({ value: child, expanded: false });
		}
		this.grad = 1;
		for (let i = topo.length - 1; i >= 0; i -= 1) {
			const value = topo[i];
			for (let j = 0; j < value.children.length; j += 1) {
				const child = value.children[j];
				child.grad = child.grad + value.localGrads[j] * value.grad;
			}
		}
	}
}

type Matrix = Value[][];
const rng = new Rng(42);
const inputPath = process.argv.length > 2 ? process.argv[2] : "../input.txt";
const docs = readFileSync(inputPath, "utf8").split(/\r?\n/).filter((line) => line.length > 0);
rng.shuffle(docs);
console.log(`num docs: ${docs.length}`);

const chars = [...new Set(docs.join("").split(""))].sort();
const BOS = chars.length;
const vocabSize = chars.length + 1;
console.log(`vocab size: ${vocabSize}`);

const nLayer = 1;
const nEmbd = 16;
const blockSize = 16;
const nHead = 4;
const headDim = nEmbd / nHead;

function matrix(nout: number, nin: number, std = 0.08): Matrix {
	return Array.from({ length: nout }, () => Array.from({ length: nin }, () => new Value(rng.gauss(0, std))));
}

const state: Record<string, Matrix> = {
	wte: matrix(vocabSize, nEmbd),
	wpe: matrix(blockSize, nEmbd),
	lmHead: matrix(vocabSize, nEmbd),
};
for (let i = 0; i < nLayer; i += 1) {
	state[`layer${i}.attnWq`] = matrix(nEmbd, nEmbd);
	state[`layer${i}.attnWk`] = matrix(nEmbd, nEmbd);
	state[`layer${i}.attnWv`] = matrix(nEmbd, nEmbd);
	state[`layer${i}.attnWo`] = matrix(nEmbd, nEmbd);
	state[`layer${i}.mlpFc1`] = matrix(4 * nEmbd, nEmbd);
	state[`layer${i}.mlpFc2`] = matrix(nEmbd, 4 * nEmbd);
}
const params: Value[] = [];
for (const matrixValue of Object.values(state)) for (const row of matrixValue) for (const value of row) params.push(value);
console.log(`num params: ${params.length}`);

function addAll(values: Value[]): Value {
	let result = new Value(0);
	for (const value of values) result = result.add(value);
	return result;
}

function linear(x: Value[], weights: Matrix): Value[] {
	return weights.map((row) => addAll(row.map((weight, i) => weight.mul(x[i]))));
}

function softmax(logits: Value[]): Value[] {
	const maxValue = Math.max(...logits.map((value) => value.data));
	const exps = logits.map((value) => value.add(-maxValue).exp());
	const total = addAll(exps);
	return exps.map((value) => value.mul(total.pow(-1)));
}

function rmsnorm(x: Value[]): Value[] {
	const meanSquare = addAll(x.map((value) => value.mul(value))).mul(1 / x.length);
	const scale = meanSquare.add(1e-5).pow(-0.5);
	return x.map((value) => value.mul(scale));
}

function gpt(tokenId: number, posId: number, keys: Value[][][], values: Value[][][]): Value[] {
	let x = state.wte[tokenId].map((token, i) => token.add(state.wpe[posId][i]));
	x = rmsnorm(x);
	for (let layer = 0; layer < nLayer; layer += 1) {
		const residual = x;
		x = rmsnorm(x);
		const q = linear(x, state[`layer${layer}.attnWq`]);
		const k = linear(x, state[`layer${layer}.attnWk`]);
		const v = linear(x, state[`layer${layer}.attnWv`]);
		keys[layer].push(k);
		values[layer].push(v);
		const attention: Value[] = [];
		for (let head = 0; head < nHead; head += 1) {
			const start = head * headDim;
			const qh = q.slice(start, start + headDim);
			const logits = keys[layer].map((past) => addAll(qh.map((qv, j) => qv.mul(past.slice(start, start + headDim)[j]))).mul(1 / nativeSqrt(headDim)));
			const weights = softmax(logits);
			for (let j = 0; j < headDim; j += 1) attention.push(addAll(weights.map((weight, t) => weight.mul(values[layer][t].slice(start, start + headDim)[j]))));
		}
		x = linear(attention, state[`layer${layer}.attnWo`]).map((value, i) => value.add(residual[i]));
		const mlpResidual = x;
		x = rmsnorm(x);
		x = linear(x, state[`layer${layer}.mlpFc1`]).map((value) => value.relu());
		x = linear(x, state[`layer${layer}.mlpFc2`]).map((value, i) => value.add(mlpResidual[i]));
	}
	return linear(x, state.lmHead);
}

const learningRate = 0.01;
const beta1 = 0.85;
const beta2 = 0.99;
const epsAdam = 1e-8;
const moments = Array.from({ length: params.length }, () => 0);
const velocities = Array.from({ length: params.length }, () => 0);
const stepEnv = process.env.MICROGPT_STEPS;
const numSteps = stepEnv === undefined ? 1000 : parseInt(stepEnv, 10);

for (let step = 0; step < numSteps; step += 1) {
	const doc = docs[step % docs.length];
	const tokens = [BOS, ...doc.split("").map((char) => chars.indexOf(char)), BOS];
	const n = Math.min(blockSize, tokens.length - 1);
	const keys = Array.from({ length: nLayer }, () => [] as Value[][]);
	const values = Array.from({ length: nLayer }, () => [] as Value[][]);
	const losses: Value[] = [];
	for (let pos = 0; pos < n; pos += 1) {
		const probabilities = softmax(gpt(tokens[pos], pos, keys, values));
		losses.push(probabilities[tokens[pos + 1]].log().neg());
	}
	const loss = addAll(losses).mul(1 / n);
	loss.backward();
	const lr = learningRate * (1 - step / numSteps);
	for (let i = 0; i < params.length; i += 1) {
		const parameter = params[i];
		moments[i] = beta1 * moments[i] + (1 - beta1) * parameter.grad;
		velocities[i] = beta2 * velocities[i] + (1 - beta2) * parameter.grad ** 2;
		const mHat = moments[i] / (1 - beta1 ** (step + 1));
		const vHat = velocities[i] / (1 - beta2 ** (step + 1));
		parameter.data -= lr * mHat / (nativeSqrt(vHat) + epsAdam);
		parameter.grad = 0;
	}
	if (step === 0 || (step + 1) % 100 === 0 || step + 1 === numSteps) console.log(`step ${String(step + 1).padStart(4)} / ${String(numSteps).padStart(4)} | loss ${loss.data.toFixed(4)}`);
}

console.log("--- inference (new, hallucinated names) ---");
const temperature = 0.5;
for (let sampleIndex = 0; sampleIndex < 20; sampleIndex += 1) {
	const keys = Array.from({ length: nLayer }, () => [] as Value[][]);
	const values = Array.from({ length: nLayer }, () => [] as Value[][]);
	let token = BOS;
	let sample = "";
	for (let pos = 0; pos < blockSize; pos += 1) {
		const logits = gpt(token, pos, keys, values).map((value) => value.mul(1 / temperature));
		token = rng.choices(softmax(logits).map((value) => value.data));
		if (token === BOS) break;
		sample += chars[token];
	}
	console.log(`sample ${String(sampleIndex + 1).padStart(2)}: ${sample}`);
}
