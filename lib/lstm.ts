import { RandMat } from "./rand-mat";
import { Mat } from "./mat";
import { Graph } from "./graph";

export interface ILstmModelLayer {
  Wix: Mat;
  Wih: Mat;
  bi: Mat;
  Wfx: Mat;
  Wfh: Mat;
  bf: Mat;
  Wox: Mat;
  Woh: Mat;
  bo: Mat;
  Wcx: Mat;
  Wch: Mat;
  bc: Mat;
}

export interface ILstmModel {
  layers: ILstmModelLayer[];
  Whd: Mat;
  bd: Mat;
}

export interface ILSTMCell {
  h: Mat[];
  c: Mat[];
  o: Mat;
}

export class LSTM {
  model: ILstmModel;
  constructor(inputSize: number, hiddenLayers: number[], outputSize: number) {
    const layers: ILstmModelLayer[] = [];
    for (let i = 0; i < hiddenLayers.length; i++) { // loop over depths
      const prevSize = i === 0 ? inputSize : hiddenLayers[i - 1];
      const hiddenSize = hiddenLayers[i];
      layers.push({
        // gates parameters
        Wix: new RandMat(hiddenSize, prevSize , 0, 0.08),
        Wih: new RandMat(hiddenSize, hiddenSize , 0, 0.08),
        bi: new Mat(hiddenSize, 1),
        Wfx: new RandMat(hiddenSize, prevSize , 0, 0.08),
        Wfh: new RandMat(hiddenSize, hiddenSize , 0, 0.08),
        bf: new Mat(hiddenSize, 1),
        Wox: new RandMat(hiddenSize, prevSize , 0, 0.08),
        Woh: new RandMat(hiddenSize, hiddenSize , 0, 0.08),
        bo: new Mat(hiddenSize, 1),
        // cell write params
        Wcx: new RandMat(hiddenSize, prevSize , 0, 0.08),
        Wch: new RandMat(hiddenSize, hiddenSize , 0, 0.08),
        bc: new Mat(hiddenSize, 1),
      });
    }
    this.model = {
      layers,
      Whd: new RandMat(outputSize, layers.length - 1, 0, 0.08),
      bd: new Mat(outputSize, 1),
    };
  }

  forward(G: Graph, hiddenLayers: number[], x: Mat, prev: null | ILSTMCell): ILSTMCell {
    const { model } = this;
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    let hiddenPrevs: Mat[];
    let cellPrevs: Mat[];
    if (prev === null || typeof prev.h === 'undefined') {
      hiddenPrevs = [];
      cellPrevs = [];
      for (let d=0;d<hiddenLayers.length;d++) {
        hiddenPrevs.push(new Mat(hiddenLayers[d],1));
        cellPrevs.push(new Mat(hiddenLayers[d],1));
      }
    } else {
      hiddenPrevs = prev.h;
      cellPrevs = prev.c;
    }

    const hidden = [];
    const cell = [];
    for (let d = 0 ; d < hiddenLayers.length; d++) {
      const layer = model.layers[d];
      const inputVector = d === 0 ? x : hidden[d-1];
      const hiddenPrev = hiddenPrevs[d];
      const cellPrev = cellPrevs[d];

      // input gate
      const h0 = G.mul(layer.Wix, inputVector);
      const h1 = G.mul(layer.Wih, hiddenPrev);
      const inputGate = G.sigmoid(G.add(G.add(h0,h1),layer.bi));

      // forget gate
      const h2 = G.mul(layer.Wfx, inputVector);
      const h3 = G.mul(layer.Wfh, hiddenPrev);
      const forgetGate = G.sigmoid(G.add(G.add(h2, h3),layer.bf));

      // output gate
      const h4 = G.mul(layer.Wox, inputVector);
      const h5 = G.mul(layer.Woh, hiddenPrev);
      const outputGate = G.sigmoid(G.add(G.add(h4, h5),layer.bo));

      // write operation on cells
      const h6 = G.mul(layer.Wcx, inputVector);
      const h7 = G.mul(layer.Wch, hiddenPrev);
      const cellWrite = G.tanh(G.add(G.add(h6, h7),layer.bc));

      // compute new cell activation
      const retainCell = G.eltmul(forgetGate, cellPrev); // what do we keep from cell
      const writeCell = G.eltmul(inputGate, cellWrite); // what do we write to cell
      const cellD = G.add(retainCell, writeCell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      const hiddenD = G.eltmul(outputGate, G.tanh(cellD));

      hidden.push(hiddenD);
      cell.push(cellD);
    }

    // one decoder to outputs at end
    const output = G.add(G.mul(this.model.Whd, hidden[hidden.length - 1]),this.model.bd);

    // return cell memory, hidden representation and output
    return {
      h: hidden,
      c: cell,
      o: output,
    };
  }

  update(alpha: number) {
    const { layers, Whd, bd } = this.model;
    for (let d = 0; d < layers.length; d++) { // loop over depths
      const layer = layers[d];
      layer.Wix.update(alpha);
      layer.Wih.update(alpha);
      layer.bi.update(alpha);
      layer.Wfx.update(alpha);
      layer.Wfh.update(alpha);
      layer.bf.update(alpha);
      layer.Wox.update(alpha);
      layer.Woh.update(alpha);
      layer.bo.update(alpha);
      layer.Wcx.update(alpha);
      layer.Wch.update(alpha);
      layer.bc.update(alpha);
    }
    Whd.update(alpha);
    bd.update(alpha);
  }
}
