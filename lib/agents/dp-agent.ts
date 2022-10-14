import { sampleWeighted } from "../utilities";

export interface IDPAgentOptions {
  gamma?: number;
  numStates: number;
  maxNumActions: number;
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
  numStates: number;
  maxNumActions: number;

  constructor(opt: IDPAgentOptions) {
    this.gamma = opt.gamma ?? 0.75; // future reward discount factor
    // reset the agent's policy and value function
    this.numStates = opt.numStates;
    this.maxNumActions = opt.maxNumActions;
    this.V = new Float64Array(this.numStates); // state value function
    this.P = new Float64Array(this.numStates * this.maxNumActions); // policy distribution \pi(s,a)
    // initialize uniform random policy
    for(let s = 0; s < this.numStates; s++) {
      const poss = this.allowedActions(s);
      for(let i = 0, n = poss.length; i < n; i++) {
        this.P[poss[i] * this.numStates + s] = 1.0 / poss.length;
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
    for(let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i];
      const prob = this.P[a * this.numStates + s];
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
    const Vnew = new Float64Array(this.numStates);
    for (let s = 0; s < this.numStates; s++) {
      // integrate over actions in a stochastic policy
      // note that we assume that policy probability mass over allowed actions sums to one
      let v = 0.0;
      const poss = this.allowedActions(s);
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        const prob = this.P[a * this.numStates + s]; // probability of taking action under policy
        if (prob === 0) { continue; } // no contribution, skip for speed
        const ns = this.nextStateDistribution(s, a);
        const rs = this.reward(s, a, ns); // reward for s->a->ns transition
        v += prob * (rs + this.gamma * this.V[ns]);
      }
      Vnew[s] = v;
    }
    this.V = Vnew; // swap
  }

  updatePolicy(): void {
    // update policy to be greedy w.r.t. learned Value function
    for (let s = 0; s < this.numStates; s++) {
      const poss = this.allowedActions(s);
      // compute value of taking each allowed action
      let vmax: number = 0;
      let nmax: number = 1;
      const vs = [];
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        const ns = this.nextStateDistribution(s,a);
        const rs = this.reward(s,a,ns);
        const v = rs + this.gamma * this.V[ns];
        vs.push(v);
        if (i === 0 || v > vmax) {
          vmax = v;
          nmax = 1;
        } else if (v === vmax) {
          nmax += 1;
        }
      }
      // update policy smoothly across all argmaxy actions
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        this.P[a*this.numStates+s] = (vs[i] === vmax) ? 1.0/nmax : 0.0;
      }
    }
  }
}
