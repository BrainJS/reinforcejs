const path = require('path');

const web = {
  target: 'web',
  entry: './lib/index.ts',
  devtool: 'source-map',
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    compress: true,
    port: 9000
  },
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.ts'],
  },
  output: {
    filename: 'rs-browser.js',
    path: path.resolve(__dirname, 'dist'),
    libraryTarget: 'umd',
  },
};

const node = {
  ...web,
  target: 'node',
  output: {
    filename: 'rs.js',
    path: path.resolve(__dirname, 'dist'),
    libraryTarget: 'umd',
  }
};

module.exports = [web, node];
