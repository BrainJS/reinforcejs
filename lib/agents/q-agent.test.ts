import {QAgent} from "./q-agent";

describe("QAgent", () => {
  class ExtendedQAgent extends QAgent {
    allowedActions(s: number): number[] {
      return [];
    }
  }
  describe("toJSON", () => {
    it("should be correct shape", () => {
      const agent = new ExtendedQAgent({
        update: "qlearn",
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        smoothPolicyUpdate: false,
        beta: 4,
        lambda: 5,
        replacingTraces: true,
        qInitVal: 6,
        planN: 7,
        inputSize: 8,
        outputSize: 9,
        P: [10],
        e: [11],
        envModelR: [12],
        pq: [13],
      });
      const json = agent.toJSON();
      expect(json).toEqual({
        update: "qlearn",
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        smoothPolicyUpdate: false,
        beta: 4,
        lambda: 5,
        replacingTraces: true,
        qInitVal: 6,
        planN: 7,
        inputSize: 8,
        outputSize: 9,
        P: [10],
        e: [11],
        envModelR: [12],
        pq: [13],
      });
    });
  });
  describe("constructor", () => {
    it("can serialize from json", () => {
      const agent = new ExtendedQAgent({
        update: "qlearn",
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        smoothPolicyUpdate: false,
        beta: 4,
        lambda: 5,
        replacingTraces: true,
        qInitVal: 6,
        planN: 7,
        inputSize: 8,
        outputSize: 9,
        P: [10],
        e: [11],
        envModelR: [12],
        pq: [13],
      });
      expect(new ExtendedQAgent(agent.toJSON())).toMatchObject(agent);
    });
  });
});
