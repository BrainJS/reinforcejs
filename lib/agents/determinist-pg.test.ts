import {DeterministPG} from "./determinist-pg";

describe("DeterministPG", () => {
  describe("toJSON", () => {
    it("should be correct shape", () => {
      const agent = new DeterministPG({
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        beta: 4,
        hiddenLayers: [5],
        inputSize: 6,
        outputSize: 7,
        activation: "sigmoid",
      });
      const json = agent.toJSON();
      expect(json).toEqual({
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        beta: 4,
        hiddenLayers: [5],
        inputSize: 6,
        outputSize: 7,
        activation: "sigmoid",
        net: agent.actorNet.toJSON(),
        criticw: agent.criticw.toJSON(),
      });
    });
  });
  describe("constructor", () => {
    it("can serialize from json", () => {
      const agent = new DeterministPG({
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        beta: 4,
        hiddenLayers: [5],
        inputSize: 6,
        outputSize: 7,
        activation: "sigmoid",
      });
      expect(new DeterministPG(agent.toJSON()).toJSON()).toMatchObject(agent.toJSON());
    });
  });
});
