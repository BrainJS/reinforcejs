import { randi } from "../utilities";
import { Mat } from "../mat";
import { Tuple } from "../tuple";
import { Graph } from "../graph";
import { INetJSON, Net } from "../net";

export interface IDQNAgentJSON {
  nh: number;
  ns: number;
  na: number;
  net: INetJSON;
}

export interface IDQNAgentOptions {
  gamma?: number;
  epsilon?: number;
  alpha?: number;
  experienceAddEvery?: number;
  experienceSize?: number;
  learningStepsPerIteration?: number;
  tderrorClamp?: number;
  numHiddenUnits?: number;
  numStates?: number;
  maxNumActions?: number;
}

export class DQNAgent {
  gamma: number;
  epsilon: number;
  alpha: number;
  experienceAddEvery: number;
  experienceSize: number;
  learningStepsPerIteration: number;
  tderrorClamp: number;
  numHiddenUnits: number;
  numStates: number;
  maxNumActions: number;
  net: Net;
  exp: Array<Tuple>;
  expi: number;

  t: number;

  r0: null | number = null;
  s0: null | Mat = null;
  s1: null | Mat = null;
  a0: null | number = null;
  a1: null | number = null;

  tderror: number = Infinity;
  lastG?: Graph;

  constructor(opt: IDQNAgentOptions = {}) {
    this.gamma = opt.gamma ?? 0.75; // future reward discount factor
    this.epsilon = opt.epsilon ?? 0.1; // for epsilon-greedy policy
    this.alpha = opt.alpha ?? 0.01; // value function learning rate
    this.experienceAddEvery = opt.experienceAddEvery ?? 25; // number of time steps before we add another experience to replay memory
    this.experienceSize = opt.experienceSize ?? 5000; // size of experience replay
    this.learningStepsPerIteration = opt.learningStepsPerIteration ?? 10;
    this.tderrorClamp = opt.tderrorClamp ?? 1.0;
    this.numHiddenUnits = opt.numHiddenUnits ?? 100;
    this.numStates = opt.numStates ?? 100;
    this.maxNumActions = opt.maxNumActions ?? 100;

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    this.net = new Net(this.numHiddenUnits, this.numStates, this.maxNumActions);

    this.exp = []; // experience
    this.expi = 0; // where to insert

    this.t = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;

    this.tderror = 0; // for visualization only...
  }
  toJSON(): IDQNAgentJSON {
    // save function
    const { numHiddenUnits, numStates, maxNumActions } = this;
    return {
      nh: numHiddenUnits,
      ns: numStates,
      na: maxNumActions,
      net: this.net.toJSON(),
    };
  }
  fromJSON(j: IDQNAgentJSON): void {
    // load function
    this.numHiddenUnits = j.nh;
    this.numStates = j.ns;
    this.maxNumActions = j.na;
    this.net = Net.fromJSON(j.net);
  }
  forwardQ(s: Mat, needs_backprop: boolean): Mat {
    const { net } = this;
    const G = new Graph(needs_backprop);
    const a1mat = G.add(G.mul(net.W1, s), net.b1);
    const h1mat = G.tanh(a1mat);
    const a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
    this.lastG = G; // back this up. Kind of hacky isn't it
    return a2mat;
  }
  act(slist: number[] | Float64Array): number {
    // convert to a Mat column vector
    const s = new Mat(this.numStates, 1);
    s.setFrom(slist);

    let a: number;
    // epsilon greedy policy
    if (Math.random() < this.epsilon) {
      a = randi(0, this.maxNumActions);
    } else {
      // greedy wrt Q function
      const amat = this.forwardQ(s, false);
      a = amat.maxi(); // returns index of argmax action
    }

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  }
  learn(r1: number): void {
    // perform an update on Q function
    if (this.r0 !== null && this.alpha > 0) {

      // learn from this tuple to get a sense of how "surprising" it is to the agent
      const t = new Tuple(this.s0 as Mat, this.a0 as number, this.r0 as number, this.s1 as Mat, this.a1 as number);
      this.tderror = this.learnFromTuple(t); // a measure of surprise

      // decide if we should keep this experience in the replay
      if (this.t % this.experienceAddEvery === 0) {
        this.exp[this.expi] = t;
        this.expi += 1;
        // roll over when we run out
        if (this.expi > this.experienceSize) {
          this.expi = 0;
        }
      }
      this.t += 1;

      // sample some additional experience from replay memory and learn from it
      for(let k = 0; k < this.learningStepsPerIteration; k++) {
        const ri = randi(0, this.exp.length); // todo: priority sweeps?
        this.learnFromTuple(this.exp[ri]);
      }
    }
    this.r0 = r1; // store for next update
  }
  learnFromTuple(t: Tuple): number {
    const { s0, a0, r0, s1, a1 } = t;
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')

    // compute the target Q value
    const tmat = this.forwardQ(s1, false);
    const i = tmat.maxi();
    const qmax = r0 + this.gamma * tmat.w[i];

    // now predict
    const pred = this.forwardQ(s0, true);

    let tderror = pred.w[a0] - qmax;
    const clamp = this.tderrorClamp;
    if (Math.abs(tderror) > clamp) {  // huber loss to robustify
      if (tderror > clamp) tderror = clamp;
      if (tderror < -clamp) tderror = -clamp;
    }
    pred.dw[a0] = tderror;
    (this.lastG as Graph).backward(); // compute gradients on net params

    // update net
    this.net.update(this.alpha);
    return tderror;
  }
}
