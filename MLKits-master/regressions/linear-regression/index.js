require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const plot = require('node-remote-plot')

const LinearRegression = require('.')

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower', 'weight', 'displacement'],
	labelColumns: ['mpg']
})

const regression = new LinearRegression(features, lables, {
	learningRate: 0.1,
	iterations: 100,
	batchSize: 10
})

regression.train()
const r2 = regression.test(testFeatures, testLabels)

plot({
	x: regression.mseHistory.reverse(),
	xLabel: 'Iteration Number',
	yLabel: 'MSE'
})

regression
	.predict([
		[120, 2, 380],
		[135, 2.1, 420]
	])
	.print()
// const mseHistory = []

// for (let b = 0; b < 500; b++) {
// 	mseHistory.push({
// 		b: b,
// 		mse: mse(b)
// 	})
// }

// const goodGuessForB = _.min(mseHistory)
