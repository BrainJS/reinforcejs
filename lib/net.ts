import {RandMat} from "./rand-mat";
import {IMatJSON, Mat} from "./mat";

export interface INetJSON {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  weights: IMatJSON[];
  biases: IMatJSON[];
}
export interface INetJSONLegacy {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
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
    public outputSize: number
  ) {
    for (let i = 0; i < hiddenLayers.length; i++) {
      const prevSize = i === 0 ? inputSize : hiddenLayers[i - 1];
      const hiddenSize = hiddenLayers[i];
      this.weights.push(new RandMat(hiddenSize, prevSize, 0, 0.01));
      this.biases.push(new Mat(hiddenSize, 1/*, 0, 0.01*/));
    }

    this.weights.push(new RandMat(outputSize, hiddenLayers[hiddenLayers.length - 1], 0, 0.1));
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
    // const n = this.W1.dw.length + this.b1.dw.length + this.W2.dw.length + this.b2.dw.length;
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
    }
  }

  static fromJSON(json: NetJSON): Net {
    const net = new Net(json.inputSize, json.hiddenLayers, json.outputSize);
    net.weights = [];
    net.biases = [];
    if (
      json.hasOwnProperty("weights") && json.hasOwnProperty("biases")) {
      net.weights = (json as INetJSON).weights.map(json => Mat.fromJSON(json))
      net.biases = (json as INetJSON).biases.map(json => Mat.fromJSON(json))
    } else {
      net.weights.push(Mat.fromJSON((json as INetJSONLegacy).W1));
      net.biases.push(Mat.fromJSON((json as INetJSONLegacy).b1));
      net.weights.push(Mat.fromJSON((json as INetJSONLegacy).W2));
      net.biases.push(Mat.fromJSON((json as INetJSONLegacy).b2));
    }
    return net;
  }
}
