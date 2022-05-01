import ezvertex as ez

ds = ez.load('simple.csv')
ds.show()

# x    y
# 1.0  4.0
# 2.0  4.6
# 4.0  5.4
# 5.0  6.0
# 6.0  6.3

ds.plot()

# shows matplotlib scatterplot of ds

ms = ez.signature(output=ds.column('y'),inputs=ds.columns('x'))
mc = ez.model_class('linear_regression')
m = ez.learn_model(mc,ms)

m.show()

# linear_regression: y = 3.5 + 0.5 * x + Gaussian(mu=0,sdev=0.1)

m.plot()

# plot of the above graph

m.predict(3)

# 5.0

ds = ez.load('census.bq')

ds.show()

# short report that it has 25 million rows and 72 columns

output = ds.column('income')
ms = ez.signature(output,ds.all_except(output))

mc = ez.automl(ms)

m = ez.learn_model(mc,ms)

ep = m.deploy(<server characteristics>)




