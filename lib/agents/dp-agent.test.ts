import {DPAgent} from "./dp-agent";

describe("DPAgent", () => {
  class ExtendedDPAgent extends DPAgent {
    allowedActions(s: number): number[] {
      return [];
    }
    nextStateDistribution(s: number, a: number): number {
      return 0;
    }
    reward(s: number, a: number, ns: number): number {
      return 0;
    }
  }
  describe("toJSON", () => {
    it("should be correct shape", () => {
      const agent = new ExtendedDPAgent({
        gamma: 1,
        inputSize: 2,
        outputSize: 3,
        V: [4],
      });
      const json = agent.toJSON();
      expect(json).toEqual({
        gamma: 1,
        inputSize: 2,
        outputSize: 3,
        V: [4],
      });
    });
  });
  describe("constructor", () => {
    it("can serialize from json", () => {
      const agent = new ExtendedDPAgent({
        gamma: 1,
        inputSize: 2,
        outputSize: 3,
        V: [4],
      });
      expect(new ExtendedDPAgent(agent.toJSON())).toMatchObject(agent);
    });
  });
});
