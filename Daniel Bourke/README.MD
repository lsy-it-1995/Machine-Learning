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
	* hidden layers: "relu"
	* output: "sigmoid"
	
loss="binary_crosentropy"=tf.keras.losses.BinaryCrossentropy()
