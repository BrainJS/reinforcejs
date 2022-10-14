import { Mat } from "../mat";
import { RandMat } from "../rand-mat";
import { Graph } from "../graph";
import { randn } from "../utilities";
import { Net } from "../net";

export interface IDeterministPGOptions {
  gamma?: number;
  epsilon?: number;
  alpha?: number;
  beta?: number;
  numStates: number;
  maxNumActions: number;
}
// Currently buggy implementation, doesnt work
export abstract class DeterministPG {
  gamma: number;
  epsilon: number;
  alpha: number;
  beta: number;
  numStates: number;
  maxNumActions: number;
  nh: number;
  actorNet: Net;
  ntheta: number;
  criticw: Mat;
  tderror: number = Infinity;
  r0: null | number = null;
  s0: null | Mat;
  s1: null | Mat;
  a0: null | Mat;
  a1: null | Mat;
  t: number;

  constructor(opt: IDeterministPGOptions) {
    this.gamma = opt.gamma ?? 0.5; // future reward discount factor
    this.epsilon = opt.epsilon ?? 0.5; // for epsilon-greedy policy
    this.alpha = opt.alpha ?? 0.001; // actor net learning rate
    this.beta = opt.beta ?? 0.01; // baseline net learning rate
    this.numStates = opt.numStates;
    this.maxNumActions = opt.maxNumActions;
    this.nh = 100; // number of hidden units

    // actor
    this.actorNet = new Net(this.nh, this.numStates, this.maxNumActions);
    this.ntheta = this.maxNumActions * this.numStates + this.maxNumActions; // number of params in actor

    // critic
    this.criticw = new RandMat(1, this.ntheta, 0, 0.01); // row vector

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
    this.t = 0;
  }

  forwardActor(s: null | Mat, needs_backprop: boolean) {
    const net = this.actorNet;
    const G = new Graph(needs_backprop);
    const a1mat = G.add(G.mul(net.W1, s as Mat), net.b1);
    const h1mat = G.tanh(a1mat);
    const a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
    return { a: a2mat, G };
  }

  act(slist: number[] | Float64Array): Mat {
    // convert to a Mat column vector
    const s = new Mat(this.numStates, 1);
    s.setFrom(slist);

    // forward the actor to get action output
    const ans = this.forwardActor(s, false);
    const amat = ans.a;
    // const ag = ans.G; TODO: This doesn't seem used

    // sample action from the stochastic gaussian policy
    const a = amat.clone();
    if (Math.random() < this.epsilon) {
      const gaussVar = 0.02;
      a.w[0] = randn(0, gaussVar);
      a.w[1] = randn(0, gaussVar);
    }
    let clamp = 0.25;
    if(a.w[0] > clamp) a.w[0] = clamp;
    if(a.w[0] < -clamp) a.w[0] = -clamp;
    if(a.w[1] > clamp) a.w[1] = clamp;
    if(a.w[1] < -clamp) a.w[1] = -clamp;

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  }

  utilJacobianAt(s: null | Mat): Mat {
    const ujacobian = new Mat(this.ntheta, this.maxNumActions);
    for (let a = 0; a < this.maxNumActions; a++) {
      this.actorNet.zeroGrads();
      const ag = this.forwardActor(this.s0, true);
      ag.a.dw[a] = 1.0;
      ag.G.backward();
      const gflat = this.actorNet.flattenGrads();
      ujacobian.setColumn(gflat,a);
    }
    return ujacobian;
  }

  learn(r1: number): void {
    // perform an update on Q function
    //this.rewardHistory.push(r1);
    if(this.r0 !== null) {
      const Gtmp = new Graph(false);
      // dpg update:
      // first compute the features psi:
      // the jacobian matrix of the actor for s
      const ujacobian0 = this.utilJacobianAt(this.s0);
      // now form the features \psi(s,a)
      const psi_sa0 = Gtmp.mul(ujacobian0, this.a0 as Mat); // should be [this.ntheta x 1] "feature" vector
      const qw0 = Gtmp.mul(this.criticw, psi_sa0); // 1x1
      // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
      const ujacobian1 = this.utilJacobianAt(this.s1);
      const ag = this.forwardActor(this.s1, false);
      const psi_sa1 = Gtmp.mul(ujacobian1, ag.a);
      const qw1 = Gtmp.mul(this.criticw, psi_sa1); // 1x1
      // get the td error finally
      let tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
      if(tderror > 0.5) tderror = 0.5; // clamp
      if(tderror < -0.5) tderror = -0.5;
      this.tderror = tderror;

      // This was converted to use method updateNaturalGradient below
      // update actor policy with natural gradient
      // const net = this.actorNet;
      // let ix = 0;
      // for(const p in net) {
      //   const mat = net[p];
      //   if(net.hasOwnProperty(p)){
      //     for(let i = 0, n = mat.w.length; i < n; i++) {
      //       mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
      //       ix+=1;
      //     }
      //   }
      // }

      // This is the conversion to use method updateNaturalGradient below
      const net = this.actorNet;
      let ix = this.updateNaturalGradient(net.W1);
      ix = this.updateNaturalGradient(net.b1, ix);
      ix = this.updateNaturalGradient(net.W2, ix);
      ix = this.updateNaturalGradient(net.b2, ix);
      // end of conversion

      // update the critic parameters too
      for(let i = 0; i < this.ntheta; i++) {
        const update = this.beta * tderror * psi_sa0.w[i];
        this.criticw.w[i] += update;
      }
    }
    this.r0 = r1; // store for next update
  }

  updateNaturalGradient(mat: Mat, ix: number = 0): number {
    for(let i = 0, n = mat.w.length; i < n; i++) {
      mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
      ix += 1;
    }
    return ix;
  }
}
