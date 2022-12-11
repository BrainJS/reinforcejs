import { randn } from "../utilities";
import { ILSTMCell, LSTM } from "../lstm";
import { Graph } from "../graph";
import { Mat } from "../mat";

export interface IRecurrentReinforceAgentOption {
  gamma?: number;
  epsilon?: number;
  alpha?: number;
  beta?: number;
  inputSize: number;
  outputSize: number;
  hiddenLayers?: number[];
}

// buggy implementation as well, doesn't work
export abstract class RecurrentReinforceAgent {
  gamma: number;
  epsilon: number;
  alpha: number;
  beta: number;

  inputSize: number;
  outputSize: number;
  hiddenLayers: number[];
  hiddenLayersBaseline: number[];

  actorLSTM: LSTM;
  actorG: Graph;
  actorPrev: null | ILSTMCell;
  actorOutputs: Mat[];
  rewardHistory: number[];
  actorActions: Mat[];
  baselineLSTM: LSTM;
  baselinePrev: null | ILSTMCell;
  baselineOutputs: Mat[];
  baselineG: Graph;
  tderror: number;
  a0: null | Mat;
  r0: null | number;
  s0: null | Mat;
  s1: null | Mat;
  a1: null | Mat;
  t: number;

  constructor(opt: IRecurrentReinforceAgentOption) {
    this.tderror = Infinity;
    this.gamma = opt.gamma ?? 0.5; // future reward discount factor
    this.epsilon = opt.epsilon ?? 0.1; // for epsilon-greedy policy
    this.alpha = opt.alpha ?? 0.001; // actor net learning rate
    this.beta = opt.beta ?? 0.01; // baseline net learning rate

    this.inputSize = opt.inputSize;
    this.outputSize = opt.outputSize;

    this.hiddenLayers = opt.hiddenLayers ?? [40]; // number of hidden units
    this.hiddenLayersBaseline = opt.hiddenLayers ?? [40]; // and also in the baseline lstm

    this.actorLSTM = new LSTM(this.inputSize, this.hiddenLayers, this.outputSize);
    this.actorG = new Graph();
    this.actorPrev = null;
    this.actorOutputs = [];
    this.rewardHistory = [];
    this.actorActions = [];

    this.baselineLSTM = new LSTM(this.inputSize, this.hiddenLayersBaseline, 1);
    this.baselineG = new Graph();
    this.baselinePrev = null;
    this.baselineOutputs = [];
    this.t = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  }

  act(slist: number[] | Float64Array): Mat {
    // convert to a Mat column vector
    const s = new Mat(this.inputSize, 1);
    s.setFrom(slist);

    // forward the LSTM to get action distribution
    const actorNext = this.actorLSTM.forward(this.actorG, this.hiddenLayers, s, this.actorPrev);
    this.actorPrev = actorNext;
    const amat = actorNext.o;
    this.actorOutputs.push(amat);

    // forward the baseline LSTM
    const baselineNext = this.baselineLSTM.forward(this.baselineG, this.hiddenLayersBaseline, s, this.baselinePrev);
    this.baselinePrev = baselineNext;
    this.baselineOutputs.push(baselineNext.o);

    // sample action from actor policy
    const gaussVar = 0.05;
    const a = amat.clone();
    for (let i = 0, n = a.w.length; i < n; i++) {
      a.w[0] += randn(0, gaussVar);
      a.w[1] += randn(0, gaussVar);
    }
    this.actorActions.push(a);

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
  }

  learn (r1: number): void {
    // perform an update on Q function
    this.rewardHistory.push(r1);
    const n = this.rewardHistory.length;
    let baselineMSE = 0.0;
    let nup = 100; // what chunk of experience to take
    let nuse = 80; // what chunk to also update
    if (n >= nup) {
      // lets learn and flush
      // first: compute the sample values at all points
      const vs = [];
      for (let t = 0; t < nuse; t++) {
        let mul = 1;
        let V = 0;
        for (let t2 = t; t2 < n; t2++) {
          V += mul * this.rewardHistory[t2];
          mul *= this.gamma;
          if (mul < 1e-5) { break; } // efficiency savings
        }
        const b = this.baselineOutputs[t].w[0];
        // todo: take out the constants etc.
        for (let i = 0; i < this.outputSize; i++) {
          // [the action delta] * [the desirebility]
          let update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
          if (update > 0.1) { update = 0.1; }
          if (update < -0.1) { update = -0.1; }
          this.actorOutputs[t].dw[i] += update;
        }
        let update = - (V - b);
        if (update > 0.1) { update = 0.1; }
        if (update < 0.1) { update = -0.1; }
        this.baselineOutputs[t].dw[0] += update;
        baselineMSE += (V-b)*(V-b);
        vs.push(V);
      }
      baselineMSE /= nuse;
      this.actorG.backward(); // update params! woohoo!
      this.baselineG.backward();
      this.actorLSTM.update(this.alpha); // update actor network
      this.baselineLSTM.update(this.beta); // update baseline network

      // flush
      this.actorG = new Graph();
      this.actorPrev = null;
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];

      this.baselineG = new Graph();
      this.baselinePrev = null;
      this.baselineOutputs = [];

      this.tderror = baselineMSE;
    }
    this.t += 1;
    this.r0 = r1; // store for next update
  }
}
