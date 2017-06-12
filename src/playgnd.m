% A = [1,3; 1,3; 1,3]
% B = [2,4; 2,4; 2,4]
% X_t = [A, B] % concatenates each column as column (Use to append COL Vectors; HORIZCAT -> Gets ROW Vector 1xn)
% Y_t = [A; B] % concatenates each column as row (Use to append ROW Vectors; VERTCAT -> Gets COL Vector nx1)
% 
% C = [2,1,2]
% D = [5;6;5]
% % Z_t = [C,D] -> Error: concatenated 
% b = size(C)
% c = size(C, 2)
% s = strcat('hey', 'hi', 'hello')

b = 0;
A = [1;2;3]
B = [1,2,3,4,5]
A*B
if (any(b==A))
    s = 'yes'
end
echo off 