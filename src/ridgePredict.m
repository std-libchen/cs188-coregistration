function [Yhat, model] = ridgePredict(X, Y, idx, idxTest, options)

% train
eigvector = SR(options.ridgePredict, Y(idx)', X(idx,:)); 

% predict
Yhat = X(idxTest,:) * eigvector;

model.eigvector = eigvector;
