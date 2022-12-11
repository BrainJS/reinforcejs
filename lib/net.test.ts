import { Net } from "./net";
import { RandMat } from "./rand-mat";
import { Mat } from "./mat";

describe("Net", () => {
  describe("constructor", () => {
    it("sets props correctly", () => {
      const net = new Net(1, [2], 3);
      expect(net.weights[0] instanceof RandMat).toBeTruthy();
      expect(net.biases[0] instanceof Mat).toBeTruthy();
      expect(net.weights[1] instanceof RandMat).toBeTruthy();
      expect(net.biases[1] instanceof Mat).toBeTruthy();
    });
  });
  describe("update", () => {
    it("calls update(alpha) on weights & biases", () => {
      const net = new Net(1, [2], 3);
      const wUpdates = net.weights.map(w => jest.spyOn(w, "update"));
      const bUpdates = net.biases.map(b => jest.spyOn(b, "update"));
      const alpha = .55;
      net.update(alpha);
      wUpdates.forEach(u => expect(u).toHaveBeenCalledWith(alpha));
      bUpdates.forEach(u => expect(u).toHaveBeenCalledWith(alpha));
    });
  });
  describe("zeroGrads", () => {
    it("calls gradFillConst(0) on weights & biases", () => {
      const net = new Net(1, [2], 3);
      const wGradFillConst = net.weights.map(w => jest.spyOn(w, "gradFillConst"));
      const bGradFillConst = net.biases.map(b => jest.spyOn(b, "gradFillConst"));
      net.zeroGrads();
      wGradFillConst.forEach(u => expect(u).toHaveBeenCalledWith(0));
      bGradFillConst.forEach(u => expect(u).toHaveBeenCalledWith(0));
    });
  });
  describe("toJSON", () => {
    it("calls toJSON() and returns json", () => {
      const net = new Net(1, [2], 3);
      expect(net.toJSON()).toEqual({
        inputSize: 1,
        hiddenLayers: [2],
        outputSize: 3,
        weights: net.weights.map(w => w.toJSON()),
        biases: net.biases.map(b => b.toJSON()),
      });
    });
  });
  describe("fromJSON", () => {
    describe("when using legacy json", () => {
      it("calls fromJSON() provides equivalent net", () => {
        const net = new Net(1, [2] ,3);
        const json = {
          W1: net.weights[0].toJSON(),
          b1: net.biases[0].toJSON(),
          W2: net.weights[1].toJSON(),
          b2: net.biases[1].toJSON(),
        };
        expect(Net.fromJSON(json)).toEqual(net);
      });
    });
    describe("when using standard json", () => {
      it("calls fromJSON() provides equivalent net", () => {
        const net = new Net(1, [2] ,3);
        const json = net.toJSON();
        expect(Net.fromJSON(json)).toEqual(net);
      });
    });
  });
});
