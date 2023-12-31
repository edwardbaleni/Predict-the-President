FLAGS <- flags(flag_integer("dense_units", 10),
               flag_integer("rate", 0.01),
               flag_integer("batch", 50))


model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$dense_units, 
              activation = "relu", 
              input_shape = c(ncol(x.train.tfidf))) %>% 
  layer_dropout(rate = FLAGS$rate) %>%
  layer_dense(units = 5, 
              activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = "accuracy"
)

hist <- model %>% fit(
    x.train.tfidf, y.train.tfidf,
    epochs = 20, batch_size = FLAGS$batch,verbose = 1)