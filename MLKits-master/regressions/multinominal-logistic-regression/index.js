requestAnimationFrame('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
// const loadCSV = require('../load-csv')
const plot = require('node-remote-plot')
const _ = require('lodash')
const mnist = require('mnist-data')

// In a function so that the garbage collector cleans up mnistData
function loadData() {
	const mnistData = mnist.training(0, 60000)

	const features = mnistData.images.values.map(image => _.flatMap(image))

	const encodedLabels = mnistData.labels.values.map(label => {
		const row = new Array(10).fill(0)
		row[label] = 1
		return row
	})

	return { features, labels }
}

const { features, labels } = loadData()

const regression = new LogisticRegression(features, labels, {
	learningRate: 1,
	iterations: 80,
	batchSize: 500
})

regression.train()

const testMnistData = mnist.testing(0, 1000)
const testFeatures = testMnistData.images.calues.map(image => _.flatMap(testMnistData))
const testEncodedLabels = testMnistData.labels.values.map(label => {
	const row = new Array(10).fill(0)
	row[label] = 1
	return row
})

const accuracy = regression.test(testFeatures, testEncodedLabels)

// const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
// 	dataColumns: ['horsepower', 'displacement', 'weight'],
// 	labelColumns: ['mpg'],
// 	shuffle: true,
// 	splitTest: 50,
// 	converters: {
// 		mpg: value => {
// 			const mpg = parseFloat(value)
// 			if (mpg < 15) {
// 				return [1, 0, 0]
// 			} else if (mpg < 30) {
// 				return [0, 1, 0]
// 			} else {
// 				return [0, 0, 1]
// 			}
// 		}
// 	}
// })

// const regression = new LogisticRegression(features, labels, {
// 	learningRate: 0.5,
// 	interations: 100,
// 	batchSize: 50,
// 	decisionBoundary: 0.6
// })

// regression.train()

// regression.test(testFeatures, _.flatMap(testLabels))
