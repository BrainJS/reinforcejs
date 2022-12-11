import {sampleWeighted} from "../utilities";

export interface IDPAgentJSON {
  gamma: number;
  inputSize: number;
  outputSize: number;
  V: number[];
}

export interface IDPAgentOptions {
  gamma?: number;
  inputSize: number;
  outputSize: number;
  V?: number[];
}
// DPAgent performs Value Iteration
// - can also be used for Policy Iteration if you really wanted to
// - requires model of the environment :(
// - does not learn from experience :(
// - assumes finite MDP :(
export abstract class DPAgent {
  V: Float64Array;
  P: Float64Array;
  gamma: number;
  inputSize: number;
  outputSize: number;

  constructor(opt: IDPAgentOptions) {
    this.gamma = opt.gamma ?? 0.75; // future reward discount factor
    // reset the agent's policy and value function
    this.inputSize = opt.inputSize;
    this.outputSize = opt.outputSize;
    this.V = opt.V ? new Float64Array(opt.V) : new Float64Array(this.inputSize); // state value function
    this.P = new Float64Array(this.inputSize * this.outputSize); // policy distribution \pi(s,a)
    // initialize uniform random policy
    for (let s = 0; s < this.inputSize; s++) {
      const poss = this.allowedActions(s);
      for (let i = 0, n = poss.length; i < n; i++) {
        this.P[poss[i] * this.inputSize + s] = 1 / poss.length;
      }
    }
  }

  abstract allowedActions(s: number): number[];
  abstract nextStateDistribution(s: number, a: number): number;
  abstract reward(s: number, a: number, ns: number): number;

  act(s: number): number {
    // behave according to the learned policy
    const poss = this.allowedActions(s);
    const ps = [];
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i];
      const prob = this.P[a * this.inputSize + s];
      ps.push(prob);
    }
    const maxi = sampleWeighted(ps);
    return poss[maxi];
  }
  learn(): void {
    // perform a single round of value iteration
    this.evaluatePolicy(); // writes this.V
    this.updatePolicy(); // writes this.P
  }
  evaluatePolicy(): void {
    // perform a synchronous update of the value function
    const vNew = new Float64Array(this.inputSize);
    for (let s = 0; s < this.inputSize; s++) {
      // integrate over actions in a stochastic policy
      // note that we assume that policy probability mass over allowed actions sums to one
      let v = 0.0;
      const poss = this.allowedActions(s);
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        const prob = this.P[a * this.inputSize + s]; // probability of taking action under policy
        if (prob === 0) { continue; } // no contribution, skip for speed
        const ns = this.nextStateDistribution(s, a);
        const rs = this.reward(s, a, ns); // reward for s->a->ns transition
        v += prob * (rs + this.gamma * this.V[ns]);
      }
      vNew[s] = v;
    }
    this.V = vNew; // swap
  }
  updatePolicy(): void {
    // update policy to be greedy w.r.t. learned Value function
    for (let s = 0; s < this.inputSize; s++) {
      const poss = this.allowedActions(s);
      // compute value of taking each allowed action
      let vMax: number = 0;
      let nMax: number = 1;
      const vs = [];
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        const ns = this.nextStateDistribution(s,a);
        const rs = this.reward(s,a,ns);
        const v = rs + this.gamma * this.V[ns];
        vs.push(v);
        if (i === 0 || v > vMax) {
          vMax = v;
          nMax = 1;
        } else if (v === vMax) {
          nMax += 1;
        }
      }
      // update policy smoothly across all argmaxy actions
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        this.P[a * this.inputSize + s] = (vs[i] === vMax) ? 1 / nMax : 0.0;
      }
    }
  }
  toJSON(): IDPAgentJSON {
    return {
      gamma: this.gamma,
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      V: Array.from(this.V),
    };
  }
}
