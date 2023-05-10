import {IMatJSON, Mat} from "../mat";
import {RandMat} from "../rand-mat";
import {Graph} from "../graph";
import {Activation, randn} from "../utilities";
import {INetJSON, Net} from "../net";

export interface IDeterministPGJSON {
  gamma: number;
  epsilon: number;
  alpha: number;
  beta: number;
  hiddenLayers: number[];
  inputSize: number;
  outputSize: number;
  activation: Activation;
  net: INetJSON;
  criticw: IMatJSON;
}
export interface IDeterministPGOptions {
  gamma?: number;
  epsilon?: number;
  alpha?: number;
  beta?: number;
  inputSize: number;
  outputSize: number;
  hiddenLayers?: number[];
  activation?: Activation;
  net?: INetJSON;
  criticw?: IMatJSON;
}
// Currently buggy implementation, doesnt work
export class DeterministPG {
  gamma: number;
  epsilon: number;
  alpha: number;
  beta: number;
  inputSize: number;
  outputSize: number;
  hiddenLayers: number[];
  actorNet: Net;
  ntheta: number;
  criticw: Mat;
  tdError: number = Infinity;
  r0: null | number = null;
  s0: null | Mat;
  s1: null | Mat;
  a0: null | Mat;
  a1: null | Mat;
  t: number;

  activation: Activation;

  constructor(opt: IDeterministPGOptions) {
    this.gamma = opt.gamma ?? 0.5; // future reward discount factor
    this.epsilon = opt.epsilon ?? 0.5; // for epsilon-greedy policy
    this.alpha = opt.alpha ?? 0.001; // actor net learning rate
    this.beta = opt.beta ?? 0.01; // baseline net learning rate
    this.hiddenLayers = opt.hiddenLayers ?? [100]; // number of hidden units
    this.inputSize = opt.inputSize;
    this.outputSize = opt.outputSize;
    this.activation = opt.activation ?? "tanh";

    // actor
    this.actorNet = opt.net ? Net.fromJSON(opt.net) : new Net(this.inputSize, this.hiddenLayers, this.outputSize);
    this.ntheta = this.outputSize * this.inputSize + this.outputSize; // number of params in actor

    // critic
    this.criticw = opt.criticw ? Mat.fromJSON(opt.criticw) : new RandMat(1, this.ntheta, 0, 0.01); // row vector

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
    this.t = 0;
  }
  forwardActor(s: null | Mat, needsBackprop: boolean) {
    const net = this.actorNet;
    const G = new Graph(needsBackprop);
    const a1mat = G.add(G.mul(net.weights[0], s as Mat), net.biases[0]);
    const h1mat = G[this.activation](a1mat);
    const a2mat = G.add(G.mul(net.weights[1], h1mat), net.biases[1]);
    return { a: a2mat, G };
  }
  act(inputs: number[] | Float64Array): Mat {
    // convert to a Mat column vector
    const s = new Mat(this.inputSize, 1);
    s.setFrom(inputs);

    // forward the actor to get action output
    const ans = this.forwardActor(s, false);
    const amat = ans.a;
    // const ag = ans.G; TODO: This doesn't seem used

    // sample action from the stochastic gaussian policy
    const action = amat.clone();
    if (Math.random() < this.epsilon) {
      const gaussVar = 0.02;
      action.w[0] = randn(0, gaussVar);
      action.w[1] = randn(0, gaussVar);
    }
    let clamp = 0.25;
    if (action.w[0] > clamp) action.w[0] = clamp;
    if (action.w[0] < -clamp) action.w[0] = -clamp;
    if (action.w[1] > clamp) action.w[1] = clamp;
    if (action.w[1] < -clamp) action.w[1] = -clamp;

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = action;

    return action;
  }
  utilJacobianAt(s: null | Mat): Mat {
    const ujacobian = new Mat(this.ntheta, this.outputSize);
    for (let a = 0; a < this.outputSize; a++) {
      this.actorNet.zeroGrads();
      const ag = this.forwardActor(this.s0, true);
      ag.a.dw[a] = 1;
      ag.G.backward();
      const gflat = this.actorNet.flattenGrads();
      ujacobian.setColumn(gflat, a);
    }
    return ujacobian;
  }
  learn(reward: number): void {
    // perform an update on Q function
    //this.rewardHistory.push(r1);
    if (this.r0 !== null) {
      const Gtmp = new Graph(false);
      // dpg update:
      // first compute the features psi:
      // the jacobian matrix of the actor for s
      const ujacobian0 = this.utilJacobianAt(this.s0);
      // now form the features \psi(s,a)
      const psiAa0 = Gtmp.mul(ujacobian0, this.a0 as Mat); // should be [this.ntheta x 1] "feature" vector
      const qw0 = Gtmp.mul(this.criticw, psiAa0); // 1x1
      // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
      const ujacobian1 = this.utilJacobianAt(this.s1);
      const ag = this.forwardActor(this.s1, false);
      const psiAa1 = Gtmp.mul(ujacobian1, ag.a);
      const qw1 = Gtmp.mul(this.criticw, psiAa1); // 1x1
      // get the td error finally
      let tdError = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
      if (tdError > 0.5) tdError = 0.5; // clamp
      if (tdError < -0.5) tdError = -0.5;
      this.tdError = tdError;

      // This was converted to use method updateNaturalGradient below
      // update actor policy with natural gradient
      // const net = this.actorNet;
      // let ix = 0;
      // for(const p in net) {
      //   const mat = net[p];
      //   if (net.hasOwnProperty(p)){
      //     for(let i = 0, n = mat.w.length; i < n; i++) {
      //       mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
      //       ix+=1;
      //     }
      //   }
      // }

      // This is the conversion to use method updateNaturalGradient below
      const net = this.actorNet;
      let ix = this.updateNaturalGradient(net.weights[0]);
      ix = this.updateNaturalGradient(net.biases[0], ix);
      ix = this.updateNaturalGradient(net.weights[1], ix);
      ix = this.updateNaturalGradient(net.biases[1], ix);
      // end of conversion

      // update the critic parameters too
      for (let i = 0; i < this.ntheta; i++) {
        const update = this.beta * tdError * psiAa0.w[i];
        this.criticw.w[i] += update;
      }
    }
    this.r0 = reward; // store for next update
  }
  updateNaturalGradient(mat: Mat, ix: number = 0): number {
    for (let i = 0, n = mat.w.length; i < n; i++) {
      mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
      ix += 1;
    }
    return ix;
  }
  toJSON(): IDeterministPGJSON {
    return {
      gamma: this.gamma,
      epsilon: this.epsilon,
      alpha: this.alpha,
      beta: this.beta,
      hiddenLayers: Array.from(this.hiddenLayers),
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      activation: this.activation,
      net: this.actorNet.toJSON(),
      criticw: this.criticw.toJSON(),
    };
  }
}
