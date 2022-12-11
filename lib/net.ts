import { RandMat } from "./rand-mat";
import {IMatJson, Mat} from "./mat";

export interface INetJSON {
  W1: IMatJson;
  b1: IMatJson;
  W2: IMatJson;
  b2: IMatJson;
}

export class Net {
  W1: RandMat;
  b1: Mat;
  W2: RandMat;
  b2: Mat;

  constructor(nh: number, ns: number, na: number) {
    this.W1 = new RandMat(nh, ns, 0, 0.01);
    this.b1 = new Mat(nh, 1/*, 0, 0.01*/);
    this.W2 = new RandMat(na, nh, 0, 0.1);
    this.b2 = new Mat(na, 1/*, 0, 0.01*/);
  }

  update(alpha: number): void {
    this.W1.update(alpha);
    this.b1.update(alpha);
    this.W2.update(alpha);
    this.b2.update(alpha);
  }

  zeroGrads(): void {
    this.W1.gradFillConst(0);
    this.b1.gradFillConst(0);
    this.W2.gradFillConst(0);
    this.b2.gradFillConst(0);
  }

  flattenGrads(): Mat {
    const n = this.W1.dw.length + this.b1.dw.length + this.W2.dw.length + this.b2.dw.length;
    const g = new Mat(n, 1);
    let ix = g.flattenGrad(this.W1);
    ix = g.flattenGrad(this.W1, ix);
    ix = g.flattenGrad(this.W1, ix);
    g.flattenGrad(this.W1, ix);
    return g;
  }

  toJSON(): INetJSON {
    return {
      W1: this.W1.toJSON(),
      b1: this.b1.toJSON(),
      W2: this.W2.toJSON(),
      b2: this.b2.toJSON(),
    }
  }

  static fromJSON(json: INetJSON): Net {
    const net = new Net(0, 0, 0);
    net.W1 = RandMat.fromJSON(json.W1);
    net.b1 = Mat.fromJSON(json.b1);
    net.W2 = RandMat.fromJSON(json.W2);
    net.b2 = Mat.fromJSON(json.b2);
    return net;
  }
}
