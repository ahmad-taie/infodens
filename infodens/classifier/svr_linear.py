'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import time


class SVR_linear(Classifier):
	'''
	classdocs
	'''

	classifierName = 'Support Vector Regressor'
	C = np.logspace(-5.0, 5.0, num=10, endpoint=True, base=2)

	def evaluate(self):

		""" Overriding default evaluate"""
		y_pred = self.predict()
		mse = mean_squared_error(self.ytest, y_pred)
		mae = mean_absolute_error(self.ytest, y_pred)
		r = pearsonr(self.ytest, y_pred) 
		return mae, mse, r

	def train(self):

		tuned_parameters = [{'C': self.C}]

		print ('SVR Optimizing. This will take a while')
		start_time = time.time()
		clf = GridSearchCV(LinearSVR(), tuned_parameters,
						   n_jobs=self.threadCount, cv=5)

		clf.fit(self.Xtrain, self.ytrain)
		print('Done with Optimizing. it took ', time.time() -
			  start_time, ' seconds')

		self.model = clf.best_estimator_

	def runClassifier(self):
		""" Overriding default running"""
		all_mse = []
		all_mae = []
		all_r = []
		#pre = []; rec = []; fsc = []

		for i in range(self.n_foldCV):
			self.shuffle()
			self.splitTrainTest()
			self.train()
			mae, mse, r = self.evaluate()
			all_mse.append( mse )
			all_mae.append( mae )
			all_r.append( r )

		classifReport = "Average MAE: {0}".format( np.mean( all_mae ) )
		classifReport += "\nAverage MSE: {0}".format( np.mean( all_mse ) )
		classifReport += "\nAverage Pearson's r: {0}".format( np.mean( all_r ) )

		#classifReport += '\nAverage Precision: ' + str(np.mean(pre))
		#classifReport += '\nAverage Recall: ' + str(np.mean(rec))
		#classifReport += '\nAverage F-score: ' + str(np.mean(fsc))

		return classifReport
