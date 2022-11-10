export type Activation = "relu" | "tanh" | "sigmoid";

export function assert(condition: boolean, message = "Assertion failed") {
  // from http://stackoverflow.com/questions/15313418/javascript-assert
  if (!condition) {
    if (typeof Error !== "undefined") {
      throw new Error(message);
    }
    throw message; // Fallback
  }
}

// Random numbers utils
let return_v = false;
let v_val = 0.0;
export const gaussRandom = function(): number {
  if (return_v) {
    return_v = false;
    return v_val;
  }
  const u = 2*Math.random()-1;
  const v = 2*Math.random()-1;
  const r = u*u + v*v;
  if (r == 0 || r > 1) return gaussRandom();
  const c = Math.sqrt(-2*Math.log(r)/r);
  v_val = v*c; // cache this
  return_v = true;
  return u*c;
}
export const randf = function(a: number, b: number): number { return Math.random()*(b-a)+a; }
export const randi = function(a: number, b: number): number { return Math.floor(Math.random()*(b-a)+a); }
export const randn = function(mu: number, std: number): number { return mu+gaussRandom()*std; }

export const samplei = function(w: number[]): number {
  // sample argmax from w, assuming w are
  // probabilities that sum to one
  const r = randf(0,1);
  let x = 0.0;
  let i = 0;
  while(true) {
    x += w[i];
    if (x > r) { return i; }
    i++;
  }
  return w.length - 1; // pretty sure we should never get here?
}

export const setConst = function<T>(arr: number[] | Float64Array, c: number): void {
  for (let i = 0, n = arr.length; i < n; i++) {
    arr[i] = c;
  }
}

export const sampleWeighted = function(p: number[]): number {
  const r = Math.random();
  let c = 0.0;
  for (let i = 0, n = p.length; i < n; i++) {
    c += p[i];
    if (c >= r) {
      return i;
    }
  }
  assert(false, 'unexpected error');
  return 0;
}
