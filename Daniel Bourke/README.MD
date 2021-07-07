# For linear model
```
model = tf.keras.Sequential([
	tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
			  optimizer=tf.keras.optimizers.SGD(),
			  metrics=["mae"])
model.fit(X, y, epoch=10)
```
mae = mean absolute error
<br />
mse = mean square error

adding mulitple hidden layers

SGD(), Adam() with learning rate(lr)

<hr />

# For binary model
```
model = tf.keras.Sequential([
	tf.keras.layers.Dense(100, activiation="relu"),
	tf.keras.layers.Dense(10, activiation="relu"),
	tf.keras.layers.Dense(1, activiation="sigmoid")
])
model.compile(loss="binary_crosentropy,
			  optimizer=tf.keras.optimizers.Adam(0.001),
			  metrics=["accuracy"])
model.fit(X, y, epoch=100)
```

* activiation: 
	* hidden layers: "relu"(Rectified Linear Unit)
	* output: "sigmoid"

[Machine activation cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
* linear
* elu
* relu
* leakyrelu
* sigmoid
* tanh
* softmax

## non-linear 
```
def relu(x):
  return tf.maximum(0, x)

def sigmoid(x):
  return 1 / (1 + tf.exp(-x))
```
loss="binary_crosentropy"=tf.keras.losses.BinaryCrossentropy()


<hr />

# plot for model
```
def plot_decision_boundary(model, X, y):
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                       np.linspace(y_min, y_max))
  x_in = np.c_[xx.ravel(), yy.ravel()]
  #predict
  y_pred = model.predict(x_in)

  if len(y_pred[0]) > 1:
    print("doing multiclass classification")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  #plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
```

# Visualize
plt on the model
if the loss going down and the accuracy going up, this means the model is improving
```
history = model.fit(X,y, epochs=...)
pd.DataFrame(history.history).plot()
```

# learning rate scheduler
before you fit, declare your learning rate callbacks
```
lr_scheduler=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4* 10**(epoch/20))
history_9 = model_9.fit(X_train, y_train, epochs=100, callbacks=[lr_scheduler])
```
Find the best loss on learning rate based on plt.figure

<hr />

# Classification evaluation methods
* Metric
	* Accuracy
		* default metric for classification problems. Not the best for imbalanced classes
	* Precision
		* higher precision leads to less false positives
	* Recall
		* higher recall leads to less false negative
	* F1-score
		* combination of precision and recall, usually a good overall metric for a classification model
	* Confusion matrix
		* when comparing predictions to truth labels to see where model gets confused. Can be hard to use with large numbers of classes