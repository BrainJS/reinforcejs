import { setConst, sampleWeighted, randi } from "../utilities";

export interface IQAgentOptions {
  update?: 'qlearn' | 'sarsa';
  gamma?: number;
  epsilon?: number;
  alpha?: number;
  smooth_policy_update?: boolean;
  beta?: number;
  lambda?: number;
  replacing_traces?: boolean;
  q_init_val?: number;
  planN?: number;
  numStates: number;
  maxNumActions: number;
}
// QAgent uses TD (Q-Learning, SARSA)
// - does not require environment model :)
// - learns from experience :)
export abstract class QAgent {
  update: string;
  gamma: number;
  epsilon: number;
  alpha: number;
  smooth_policy_update: boolean;
  beta: number;
  lambda: number;
  replacing_traces: boolean;
  q_init_val: number;
  planN: number;
  Q: number[] | Float64Array;
  P: number[] | Float64Array;
  e: number[] | Float64Array;
  env_model_s: number[] | Float64Array;
  env_model_r: number[] | Float64Array;
  numStates: number;
  maxNumActions: number;
  sa_seen: number[];
  pq: number[] | Float64Array;

  constructor(opt: IQAgentOptions) {
    this.update = opt.update ?? 'qlearn'; // qlearn | sarsa
    this.gamma = opt.gamma ?? 0.75; // future reward discount factor
    this.epsilon = opt.epsilon ?? 0.1; // for epsilon-greedy policy
    this.alpha = opt.alpha ?? 0.01; // value function learning rate

    // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
    this.smooth_policy_update = opt.smooth_policy_update ?? false;
    this.beta = opt.beta ?? 0.01; // learning rate for policy, if smooth updates are on

    // eligibility traces
    this.lambda = opt.lambda ?? 0; // eligibility trace decay. 0 = no eligibility traces used
    this.replacing_traces = opt.replacing_traces ?? true;

    // optional optimistic initial values
    this.q_init_val = opt.q_init_val ?? 0;

    this.planN = opt.planN ?? 0; // number of planning steps per learning iteration (0 = no planning)

    this.numStates = opt.numStates;
    this.maxNumActions = opt.maxNumActions;

    this.Q = new Float64Array(this.numStates * this.maxNumActions); // state action value function
    if (this.q_init_val !== 0) {
      setConst(this.Q, this.q_init_val);
    }
    this.P = new Float64Array(this.numStates * this.maxNumActions); // policy distribution \pi(s,a)
    this.e = new Float64Array(this.numStates * this.maxNumActions); // eligibility trace
    this.env_model_s = new Float64Array(this.numStates * this.maxNumActions); // environment model (s,a) -> (s',r)
    setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
    this.env_model_r = new Float64Array(this.numStates * this.maxNumActions); // environment model (s,a) -> (s',r)

    this.sa_seen = [];
    this.pq = new Float64Array(this.numStates * this.maxNumActions);

    // initialize uniform random policy
    for (let s = 0; s < this.numStates; s++) {
      const poss = this.allowedActions(s);
      for(let i = 0, n = poss.length; i < n; i++) {
        this.P[poss[i] * this.numStates + s] = 1.0 / poss.length;
      }
    }
    // agent memory, needed for streaming updates
    // (s0,a0,r0,s1,a1,r1,...)
    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  }

  abstract allowedActions(s: number): number[];

  r0: null | number = null;
  s0: null | number = null;
  s1: null | number = null;
  a0: null | number = null;
  a1: null | number = null;

  resetEpisode(): void {
    // an episode finished
  }

  explored: boolean = false;

  act(s: number): number {
    // act according to epsilon greedy policy
    const poss = this.allowedActions(s);
    const probs = [];
    for(let i = 0, n = poss.length; i < n; i++) {
      probs.push(this.P[poss[i] * this.numStates + s]);
    }

    let a: number;
    // epsilon greedy policy
    if (Math.random() < this.epsilon) {
      a = poss[randi(0,poss.length)]; // random available action
      this.explored = true;
    } else {
      a = poss[sampleWeighted(probs)];
      this.explored = false;
    }
    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
  }

  learn(r1: number): void {
    // takes reward for previous action, which came from a call to act()
    if (this.r0 !== null) {
      this.learnFromTuple(this.s0 as number, this.a0 as number, this.r0 as number, this.s1 as number, this.a1 as number, this.lambda);
      if (this.planN > 0) {
        this.updateModel(this.s0 as number, this.a0 as number, this.r0 as number, this.s1 as number);
        this.plan();
      }
    }
    this.r0 = r1; // store this for next update
  }

  updateModel(s0: number, a0: number, r0: number, s1: number): void {
    // transition (s0,a0) -> (r0,s1) was observed. Update environment model
    const sa = a0 * this.numStates + s0;
    if (this.env_model_s[sa] === -1) {
      // first time we see this state action
      this.sa_seen.push(a0 * this.numStates + s0); // add as seen state
    }
    this.env_model_s[sa] = s1;
    this.env_model_r[sa] = r0;
  }

  plan(): void {
    // order the states based on current priority queue information
    const spq = [];
    for(let i = 0, n = this.sa_seen.length; i < n; i++) {
      const sa = this.sa_seen[i];
      const sap = this.pq[sa];
      if (sap > 1e-5) { // gain a bit of efficiency
        spq.push({sa:sa, p:sap});
      }
    }
    spq.sort(function(a, b){ return a.p < b.p ? 1 : -1});

    // perform the updates
    const nsteps = Math.min(this.planN, spq.length);
    for(let k = 0; k < nsteps; k++) {
      // random exploration
      //var i = randi(0, this.sa_seen.length); // pick random prev seen state action
      //var s0a0 = this.sa_seen[i];
      const s0a0 = spq[k].sa;
      this.pq[s0a0] = 0; // erase priority, since we're backing up this state
      const s0 = s0a0 % this.numStates;
      const a0 = Math.floor(s0a0 / this.numStates);
      const r0 = this.env_model_r[s0a0];
      const s1 = this.env_model_s[s0a0];
      let a1 = -1; // not used for Q learning
      if (this.update === 'sarsa') {
        // generate random action?...
        const poss = this.allowedActions(s1);
        a1 = poss[randi(0,poss.length)];
      }
      this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
    }
  }

  learnFromTuple(s0: number, a0: number, r0: number, s1: number, a1: number, lambda: number): void {
    const sa = a0 * this.numStates + s0;

    let target: number = 0;
    // calculate the target for Q(s,a)
    if (this.update === 'qlearn') {
      // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
      const poss = this.allowedActions(s1);
      let qmax = 0;
      for (let i = 0, n = poss.length; i < n; i++) {
        const s1a = poss[i] * this.numStates + s1;
        const qval = this.Q[s1a];
        if (i === 0 || qval > qmax) { qmax = qval; }
      }
      target = r0 + this.gamma * qmax;
    } else if (this.update === 'sarsa') {
      // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
      let s1a1 = a1 * this.numStates + s1;
      target = r0 + this.gamma * this.Q[s1a1];
    }

    if (lambda > 0) {
      // perform an eligibility trace update
      if (this.replacing_traces) {
        this.e[sa] = 1;
      } else {
        this.e[sa] += 1;
      }
      const edecay = lambda * this.gamma;
      const state_update = new Float64Array(this.numStates);
      for(let s = 0; s < this.numStates; s++) {
        const poss = this.allowedActions(s);
        for(let i = 0; i < poss.length; i++) {
          const a = poss[i];
          const saloop = a * this.numStates + s;
          const esa = this.e[saloop];
          const update = this.alpha * esa * (target - this.Q[saloop]);
          this.Q[saloop] += update;
          this.updatePriority(s, a, update);
          this.e[saloop] *= edecay;
          const u = Math.abs(update);
          if (u > state_update[s]) { state_update[s] = u; }
        }
      }
      for (let s = 0; s < this.numStates; s++) {
        if (state_update[s] > 1e-5) { // save efficiency here
          this.updatePolicy(s);
        }
      }
      if (this.explored && this.update === 'qlearn') {
        // have to wipe the trace since q learning is off-policy :(
        this.e = new Float64Array(this.numStates * this.maxNumActions);
      }
    } else {
      // simpler and faster update without eligibility trace
      // update Q[sa] towards it with some step size
      const update = this.alpha * (target - this.Q[sa]);
      this.Q[sa] += update;
      this.updatePriority(s0, a0, update);
      // update the policy to reflect the change (if appropriate)
      this.updatePolicy(s0);
    }
  }

  updatePriority(s: number, a: number, u: number): void {
    // used in planning. Invoked when Q[sa] += update
    // we should find all states that lead to (s,a) and upgrade their priority
    // of being update in the next planning step
    u = Math.abs(u);
    if (u < 1e-5) { return; } // for efficiency skip small updates
    if (this.planN === 0) { return; } // there is no planning to be done, skip.
    for (let si = 0; si < this.numStates; si++) {
      // note we are also iterating over impossible actions at all states,
      // but this should be okay because their env_model_s should simply be -1
      // as initialized, so they will never be predicted to point to any state
      // because they will never be observed, and hence never be added to the model
      for (let ai = 0; ai < this.maxNumActions; ai++) {
        const siai = ai * this.numStates + si;
        if (this.env_model_s[siai] === s) {
          // this state leads to s, add it to priority queue
          this.pq[siai] += u;
        }
      }
    }
  }

  updatePolicy(s: number): void {
    const poss = this.allowedActions(s);
    // set policy at s to be the action that achieves max_a Q(s,a)
    // first find the maxy Q values
    let qmax: number = 0, nmax: number = 1;
    const qs: number[] = [];
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i];
      const qval = this.Q[a * this.numStates + s];
      qs.push(qval);
      if (i === 0 || qval > qmax) {
        qmax = qval;
        nmax = 1;
      } else if (qval === qmax) {
        nmax += 1;
      }
    }
    // now update the policy smoothly towards the argmaxy actions
    let psum = 0.0;
    for (let i = 0, n = poss.length; i < n; i++) {
      const a = poss[i];
      const target = (qs[i] === qmax) ? 1.0 / nmax : 0.0;
      const ix = a*this.numStates+s;
      if (this.smooth_policy_update) {
        // slightly hacky :p
        this.P[ix] += this.beta * (target - this.P[ix]);
        psum += this.P[ix];
      } else {
        // set hard target
        this.P[ix] = target;
      }
    }
    if (this.smooth_policy_update) {
      // renomalize P if we're using smooth policy updates
      for (let i = 0, n = poss.length; i < n; i++) {
        const a = poss[i];
        this.P[a * this.numStates + s] /= psum;
      }
    }
  }
}
