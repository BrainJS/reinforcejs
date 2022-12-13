import {RandMat} from "./rand-mat";
import {IMatJSON, Mat} from "./mat";

export interface INetJSON {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  weights: IMatJSON[];
  biases: IMatJSON[];
  std: number;
}

export interface INetJSONLegacy {
  W1: IMatJSON;
  b1: IMatJSON;
  W2: IMatJSON;
  b2: IMatJSON;
}

export type NetJSON = INetJSON | INetJSONLegacy;

export class Net {
  weights: Mat[] = [];
  biases: Mat[] = [];

  constructor(
    public inputSize: number,
    public hiddenLayers: number[],
    public outputSize: number,
    public std: number = 0.01,
  ) {
    let prevSize = inputSize;
    for (let i = 0; i < hiddenLayers.length; i++) {
      const hiddenSize = hiddenLayers[i];
      this.weights.push(new RandMat(hiddenSize, prevSize, 0, this.std));
      this.biases.push(new Mat(hiddenSize, 1/*, 0, 0.01*/));
      prevSize = hiddenLayers[i];
    }

    this.weights.push(new RandMat(outputSize, prevSize, 0, this.std));
    this.biases.push(new Mat(outputSize, 1/*, 0, 0.01*/));
  }

  update(alpha: number): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i].update(alpha);
      this.biases[i].update(alpha);
    }
  }

  zeroGrads(): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i].gradFillConst(0);
      this.biases[i].gradFillConst(0);
    }
  }

  flattenGrads(): Mat {
    let n = 0;
    for (let i = 0; i < this.weights.length; i++) {
      n += this.weights[i].dw.length + this.biases[i].dw.length;
    }
    const g = new Mat(n, 1);
    let ix = g.flattenGrad(this.weights[0]);
    ix = g.flattenGrad(this.biases[0], ix);
    for (let i = 1; i < this.weights.length; i++) {
      ix = g.flattenGrad(this.weights[i], ix);
      ix = g.flattenGrad(this.biases[i], ix);
    }
    return g;
  }

  toJSON(): INetJSON {
    return {
      inputSize: this.inputSize,
      hiddenLayers: [...this.hiddenLayers],
      outputSize: this.outputSize,
      weights: this.weights.map(m => m.toJSON()),
      biases: this.biases.map(m => m.toJSON()),
      std: this.std,
    }
  }

  static fromJSON(json: NetJSON | INetJSONLegacy): Net {
    if (
      json.hasOwnProperty("weights")
      && json.hasOwnProperty("biases")
      && json.hasOwnProperty("inputSize")
      && json.hasOwnProperty("hiddenLayers")
      && json.hasOwnProperty("outputSize")
      && json.hasOwnProperty("std")
    ) {
      const standardJSON = json as INetJSON;
      const net = new Net(standardJSON.inputSize, standardJSON.hiddenLayers, standardJSON.outputSize, standardJSON.std);
      net.weights = [];
      net.biases = [];
      net.weights = standardJSON.weights.map(json => Mat.fromJSON(json))
      net.biases = standardJSON.biases.map(json => Mat.fromJSON(json))
      return net;
    } else if (
      json.hasOwnProperty("W1")
      && json.hasOwnProperty("W2")
      && json.hasOwnProperty("b1")
      && json.hasOwnProperty("b2")
    ) {
      const legacyJSON = json as INetJSONLegacy;
      const net = new Net(legacyJSON.W1.d, [legacyJSON.W1.n], legacyJSON.W2.n);
      net.weights = [];
      net.biases = [];
      net.weights.push(Mat.fromJSON(legacyJSON.W1));
      net.biases.push(Mat.fromJSON(legacyJSON.b1));
      net.weights.push(Mat.fromJSON(legacyJSON.W2));
      net.biases.push(Mat.fromJSON(legacyJSON.b2));
      return net;
    }
    throw new Error("unknown json");
  }
}
