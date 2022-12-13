// Mat holds a matrix
import {assert, randf, randn} from "./utilities";

export interface IMatJSON {
  n: number;
  d: number;
  w: number[] | { [key: number]: number };
}

export class Mat {
  w: Float64Array;
  dw: Float64Array;
  constructor(
    public n: number,
    public d: number) {
    // n is number of rows d is number of columns
    this.w = new Float64Array(n * d);
    this.dw = new Float64Array(n * d);
  }
  get(row: number, col: number): number {
    // slow but careful accessor function
    // we want row-major order
    const ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  }
  set(row: number, col: number, v: number): void {
    // slow but careful accessor function
    let ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  }
  setFrom(arr: number[] | Float64Array): void {
    for (let i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i];
    }
  }
  setColumn(m: this, i: number): void {
    for (let q = 0, n = m.w.length; q < n; q++) {
      this.w[(this.d * q) + i] = m.w[q];
    }
  }
  toJSON(): IMatJSON {
    return {
      n: this.n,
      d: this.d,
      w: Array.from(this.w),
    };
  }
  fromJSON(json: IMatJSON): this {
    this.n = json.n;
    this.d = json.d;
    this.w = new Float64Array(this.n * this.d);
    this.dw = new Float64Array(this.n * this.d);
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
    return this;
  }
  clone(): Mat {
    const a = new Mat(this.n, this.d);
    a.setFrom(this.w);
    return a;
  }
  update(alpha: number): void {
    // updates in place
    for (let i = 0, n = this.n * this.d; i < n; i++) {
      if (this.dw[i] !== 0) {
        this.w[i] += - alpha * this.dw[i];
        this.dw[i] = 0;
      }
    }
  }
  flattenGrad(mat: Mat, ix = 0): number {
    for (let i = 0, m = mat.dw.length; i < m; i++) {
      this.w[ix] = mat.dw[i];
      ix++;
    }
    return ix;
  }
  gradFillConst(c: number): void {
    for (let i = 0, n = this.dw.length; i < n; i++) {
      this.dw[i] = c;
    }
  }
  fillRandn(mu: number, std: number): void {
    for (let i = 0, n = this.w.length; i < n; i++) {
      this.w[i] = randn(mu, std);
    }
  }
  fillRand(lo: number, hi: number) {
    for (let i = 0, n = this.w.length; i < n; i++) {
      this.w[i] = randf(lo, hi);
    }
  }
  maxi(): number {
    // argmax of this.w
    const w = this.w;
    let maxv = w[0];
    let maxix = 0;
    for (let i = 1, n = w.length; i < n; i++) {
      const v = w[i];
      if (v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  }
  static fromJSON(json: IMatJSON): Mat {
    const mat = new Mat(1, 1);
    mat.fromJSON(json);
    return mat;
  }
}
