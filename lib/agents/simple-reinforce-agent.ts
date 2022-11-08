import {activation, randn} from "../utilities";
import { Mat } from "../mat";
import { Graph } from "../graph";
import { Net } from "../net";

export interface ISimpleReinforceAgentOption {
  gamma?: number;
  epsilon?: number;
  alpha?: number;
  beta?: number;
  inputSize: number;
  outputSize: number;
  activation: activation;
}

// buggy implementation, doesnt work...
export abstract class SimpleReinforceAgent {
  gamma: number;
  epsilon: number;
  alpha: number;
  beta: number;
  inputSize: number;
  outputSize: number;
  nh: number;
  nhb: number;
  actorNet: Net;
  actorOutputs: Mat[];
  actorGraphs: Graph[];
  actorActions: Mat[]; // sampled ones
  activation: activation;

  rewardHistory: number[];

  baselineNet: Net;
  baselineOutputs: Mat[];
  baselineGraphs: Graph[];
  t: number;

  r0: null | number;
  s0: null | Mat;
  a0: null | Mat;
  s1: null | Mat;
  a1: null | Mat;

  tderror: number;

  constructor(opt: ISimpleReinforceAgentOption) {
    this.gamma = opt.gamma ?? 0.5; // future reward discount factor
    this.epsilon = opt.epsilon ?? 0.75; // for epsilon-greedy policy
    this.alpha = opt.alpha ?? 0.001; // actor net learning rate
    this.beta = opt.beta ?? 0.01; // baseline net learning rate
    this.inputSize = opt.inputSize;
    this.outputSize = opt.outputSize;
    this.nh = 100; // number of hidden units
    this.nhb = 100; // and also in the baseline lstm
    this.activation = opt.activation ?? "tanh";

    this.actorNet = new Net(this.inputSize, [this.nh], this.outputSize);
    this.actorOutputs = [];
    this.actorGraphs = [];
    this.actorActions = []; // sampled ones

    this.rewardHistory = [];

    this.baselineNet = new Net(this.inputSize, [this.nhb], this.outputSize);
    this.baselineOutputs = [];
    this.baselineGraphs = [];

    this.t = 0;
    this.r0 = null;
    this.s0 = null;
    this.a0 = null;
    this.s1 = null;
    this.a1 = null;

    this.tderror = 0;
  }

  forwardActor(s: Mat, needsBackprop: boolean) {
    const net = this.actorNet;
    const G = new Graph(needsBackprop);
    const a1mat = G.add(G.mul(net.weights[0], s), net.biases[0]);
    const h1mat = G[this.activation](a1mat);
    const a2mat = G.add(G.mul(net.weights[1], h1mat), net.biases[1]);
    return {
      a: a2mat,
      G,
    };
  }

  forwardValue(s: Mat, needsBackprop: boolean) {
    const net = this.baselineNet;
    const G = new Graph(needsBackprop);
    const a1mat = G.add(G.mul(net.weights[0], s), net.biases[0]);
    const h1mat = G[this.activation](a1mat);
    const a2mat = G.add(G.mul(net.weights[1], h1mat), net.biases[1]);
    return {
      a: a2mat,
      G,
    };
  }

  act(slist: number[] | Float64Array): Mat {
    // convert to a Mat column vector
    const s = new Mat(this.inputSize, 1);
    s.setFrom(slist);

    // forward the actor to get action output
    const ans1 = this.forwardActor(s, true);
    const amat = ans1.a;
    const ag = ans1.G;
    this.actorOutputs.push(amat);
    this.actorGraphs.push(ag);

    // forward the baseline estimator
    const ans2 = this.forwardValue(s, true);
    const vmat = ans2.a;
    const vg = ans2.G;
    this.baselineOutputs.push(vmat);
    this.baselineGraphs.push(vg);

    // sample action from the stochastic gaussian policy
    const a = amat.clone();
    const gaussVar = 0.02;
    a.w[0] = randn(0, gaussVar);
    a.w[1] = randn(0, gaussVar);
    this.actorActions.push(a);

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  }

  learn(r1: number) {
    // perform an update on Q function
    this.rewardHistory.push(r1);
    const n = this.rewardHistory.length;
    let baselineMSE = 0.0;
    let nup = 100; // what chunk of experience to take
    let nuse = 80; // what chunk to update from
    if (n >= nup) {
      // lets learn and flush
      // first: compute the sample values at all points
      const vs = [];
      for (let t = 0; t < nuse; t++) {
        let mul = 1;
        // compute the actual discounted reward for this time step
        let V = 0;
        for (let t2 = t; t2 < n; t2++) {
          V += mul * this.rewardHistory[t2];
          mul *= this.gamma;
          if (mul < 1e-5) { break; } // efficiency savings
        }
        // get the predicted baseline at this time step
        const b = this.baselineOutputs[t].w[0];
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
        baselineMSE += (V - b) * (V - b);
        vs.push(V);
      }
      baselineMSE /= nuse;
      // backprop all the things
      for (let t = 0; t < nuse; t++) {
        this.actorGraphs[t].backward();
        this.baselineGraphs[t].backward();
      }
      this.actorNet.update(this.alpha); // update actor network
      this.baselineNet.update(this.beta); // update baseline network

      // flush
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];
      this.baselineOutputs = [];
      this.actorGraphs = [];
      this.baselineGraphs = [];

      this.tderror = baselineMSE;
    }
    this.t += 1;
    this.r0 = r1; // store for next update
  }
}
