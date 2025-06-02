
n=input('Enter No. of Images for training'); 
L=input('Enter No. of Dominant Eigen Values to keep ');
M=100; N=90; 
X=zeros(n,(M*N)); 
T=zeros(n,L);  
for count=1 :n 
I=imread(sprintf('%d.jpeg',count));  
I=rgb2gray(I);
I=imresize(I,[M,N]); 
X(count,:)=reshape (I, [1, M*N]);  
end
Xb=X; 
m=mean(X); 

for i=1:n 

X(i,:)=X(i,:)-m;  
end

Q=(X'*X)/(n-1);
[Evecm, Evalm]=eig(Q) ; 
Eval=diag (Evalm);  
[Evalsorted, Index]=sort (Eval, 'descend');  
Evecsorted=Evecm(:, Index); 
Ppca = Evecsorted(:,1:L);

for i=1:n 
T(i,:)=(Xb(i,:)-m)*Ppca;
end 

