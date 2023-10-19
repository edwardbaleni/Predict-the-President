FLAGS <- flags(flag_integer("dense_units", "rate"))


model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$dense_units, 
              activation = "relu", 
              input_shape = c(ncol(x.train.tfidf))) %>% 
  layer_dropout(rate = FLAGS$rate) %>%
  layer_dense(units = 5, 
              activation = 'softmax')