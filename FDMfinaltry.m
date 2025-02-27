function FDMfinaltry
xO=0;
xL=2;
yO=0;
yL=1;

n=3;      %n - mesh for x (in matrices for column,j)
m=3;       %m - mesh for y (in matrices for row,i )
dx=(xL-xO)/(n+1);
dy=(yL-yO)/(m+1);


%%
%Matrix A
for i=1:n*m
    A(i,i)=-2*(1/dx^2+1/dy^2);
end

for i=1:n-1 
    %to consturct lower and upper diagonal
    for j=1:m
    A(i+(j-1)*n, i+(j-1)*n+1)= 1/dx^2;
    A(i+(j-1)*n+1, i+(j-1)*n)= 1/dx^2;
    end
end

for i=1:n
    for j=1:m-1
 A(i+(j-1)*n,i+j*n)= 1/dy^2;
 A(i+j*n,i+(j-1)*n)= 1/dy^2;
    end
end



%%
%set up BC's (INSERT EQUATION FOR BOUNDARY EQUATIONS)

x=linspace(xO,xL,n+2);
y=linspace(yO,yL,m+2);

tle=@(x,y)0;
tr=@(x,y)2*exp(y);
tso=@(x,y)x;
tn=@(x,y)exp(1)*x;





%%
%Matrix     p
%Matrix p is the the boundary condition in the form of matrix
p = zeros(m+2,n+2);
for j=1:n+2
    p(1,j) = tso(x(1,j),y(1,1));
    p(m+2,j) =tn(x(1,j),y(1,m+2));
end
for i=1:m+2
    p(i,1) = tle(x(1,1),y(1,i));
    p(i,n+2) = tr(x(1,n+2),y(1,i));
end
p;

%%
%Matrix B
 b = zeros(n*m,1);
 for j=1:n-2
     b(j+1,1)=p(1,j+2)/dy^2;         %pink
     b(m*n-j,1)=p(m+2,n-j+1)/dy^2;    %hijau
     
     for k=1:m-2                 %update baru tukar pada m    
     b(k*n+n,1)=p(k+2,n+2)/dx^2;      %kuning tengah
     b(k*n+1,1)=p(k+2,1)/dx^2;        %oren tengah
     end 
 end
  b(1,1)=p(1,2)/dy^2+p(2,1)/dx^2;              %oren + pink
  b(n,1)=p(1,n+1)/dy^2+p(2,n+2)/dx^2;          %pink + kuning
  b(m*n-(n-1),1)=p(m+1,1)/dx^2+p(m+2,2)/dy^2;  %oren + hijau
  b(m*n,1)=p(m+2,n+1)/dy^2+p(m+1,n+2)/dx^2;    %hijau + kuning
%%
%Matrix bpoisson

 % Uxx+Uyy=f to insert equation 
f=@(x,y)x*exp(y);
bpoisson= zeros(n*m,1);
for k=n
    for i=1:m
        for j=1:n
        bpoisson(j+((i-1)*k),1)=f(x(1,j+1),y(1,i+1));%4
%         x(1,j+1)*exp(1)^ynew(i+1,1);%4*x(1,j+1)+3*ynew(i+1,1); %insert equation for poisson
        end
    end
end
bpoisson
b=bpoisson-b;

%%    
%Jacobi &  Gauss Seidel & SOR
% dxsolve = (xL-xO)/(n*m-1);
% xsolve = xO:dxsolve:xL;
xsolve=zeros(n*m,1);
xj = xsolve;
xg = xsolve;
xs = xsolve;

L =tril(-A,-1);         %triangular
D = diag(diag(A));      %diagonal A. diag(diag(A))
U = triu(-A,1);         %u triangular

Tj = inv(D)*(L+U);       %sama dengan T = inv(D)*(A-D) except -+
Cj = inv(D)*b;
Tg = inv(D-L)*(U);
Cg = inv(D-L)*b;

Ts = inv(D)*(L+U);
rho_Ts = max(abs(eig(Ts)));
 %w = 2/(1 + sqrt(1-rho_Ts^2));
 w=1.25;
Tw = inv(D-w*L)*((1-w)*D+w*U);
Cw = w*(inv(D-w*L))*b;
delta = 1e-5;          %mesti ada bila buat tolerance
N=10000000000000;
dj=zeros(21,1);  %utk simpan tolerance
jc=1;
tic %mesti ada bila buat while
while jc<=N              %N is the iteration
    xj(:,jc+1)=Tj*xj(:,jc)+Cj;     
    %tolerance
%     tolerancej = norm(xj(:,jc+1)-xj(:,jc),inf)/norm(xj(:,jc+1),inf);
   tolerancej = norm(xj(:,jc+1)-xj(:,jc),inf)
    if tolerancej<delta
        break
    end
     jc=jc+1;              
     %fprintf('n=%g tolerancej=%g\n',jc,tolerancej)
end
toc
tolerancej;
g=1;
tic 
while g<=N              %N is the iteration
    xg(:,g+1)=Tg*xg(:,g)+Cg;
    %tolerance    
    %toleranceg = norm(xg(:,g+1)-xg(:,g),inf)/norm(xg(:,g+1),inf);
    toleranceg = norm(xg(:,g+1)-xg(:,g),inf)
    if toleranceg<delta
        break
    end
    g=g+1;              
    %fprintf('n=%g toleranceg=%g\n',g,toleranceg)
end
toc
s=1;
tic
while s<=N
    xs(:,s+1) = Tw*xs(:,s)+Cw;
    %tolerance
    %tolerances = norm(xs(:,s+1)-xs(:,s),inf)/norm(xs(:,s+1),inf);
    tolerances = norm(xs(:,s+1)-xs(:,s));
    if tolerances < delta
        break
    end
    s=s+1;
    %fprintf('n=%g tolerances=%g\n',s,tolerances)
end
toc
xj = xj(:,jc+1);
xg = xg(:,g+1);
xs = xs(:,s+1)
%%
% Conjugate
%ni: number of iteration
% %%
x0 = zeros(n*m,1); %zeros x0 M*N
conditionalA=cond(A,Inf);
% ni=5
r=[];
v=[];
alfa=[];
beta=[];
Asize=size(A);
F=Asize(1,1);

ic=1;
deltac=1e-5;
for nn=1:F
    xc(nn,k)=x0(nn,1);
end
r(:,ic)=b-A*xc(:,ic);
v(:,ic)=r(:,ic);
tic
while ic<=F
    alfa(:,ic)=r(:,ic)'*r(:,ic)/(v(:,ic)'*A*v(:,ic));
    xc(:,(ic+1))=xc(:,ic)+alfa(:,ic)*v(:,ic);
    r(:,(ic+1))=r(:,ic)-alfa(:,ic)*A*v(:,ic);
    
    tolc=norm(r(:,ic+1));
%     tolc=norm(r(:,k+1)-r(:,k),inf)
    if tolc<deltac
        break;
    end
    beta(:,ic)=r(:,(ic+1))'*r(:,(ic+1))/(r(:,ic)'*r(:,ic));
    v(:,(ic+1))=r(:,(ic+1))+beta(:,ic)*v(:,ic);
    
    ic=ic+1;
    %fprintf('n=%g tolerancec=%g\n',ic,tolc)
end
toc
xc = xc(:,(ic+1))
ic;
% %%

%%
%ni: number of iteration
x0 = zeros(n*m,1); %zeros x0 M*N
conditionalA=cond(A,Inf);
r=[];
v=[];
alfa=[];
beta=[];
Anew=-1*A
Asize=size(A);
F=10000000000;
Cinver=(diag(diag(Anew)))^(-1/2)
ip=1;

deltac=1e-5;
xp(:,ip)=x0(:,ip);
r(:,ip)=b-A*xp(:,ip);
v(:,ip)=Cinver*r(:,ip);
tic
while ip<=F
    alfa(:,ip)=r(:,ip)'*Cinver*r(:,ip)/(v(:,ip)'*A*v(:,ip));
    xp(:,(ip+1))=xp(:,ip)+alfa(:,ip)*v(:,ip);
    r(:,(ip+1))=r(:,ip)-alfa(:,ip)*A*v(:,ip);
    
    tolp=norm(r(:,ip+1));
 
    if tolp<deltac
        break;
    end
    beta(:,ip)=r(:,(ip+1))'*Cinver*r(:,(ip+1))/(v(:,ip)'*A*v(:,ip));
    v(:,(ip+1))=Cinver*r(:,(ip+1))+beta(:,ip)*v(:,ip);
    
    ip=ip+1;
    
end
toc
xp = xp(:,(ip+1))
ip
%%
%Preconditioned Conjugate Gradient

x0=zeros(n*m,1);	
u = x0;
r = b - A * u;
C = (D+L)*inv(D)*(D+L');
D = diag(diag(A));
C1 = tril(A) ;
C2 = inv(D)*triu(A);
pho = C2 \ (C1 \ r);
norm_b = norm(b);
 tol=1e-5;
 m_max=100000000;
iterp = 0;
% tic
% while( (norm(r)/norm_b > tol) & (iterp < m_max))
%   a = A * pho;
%   a_dot_p = a' * pho;
%   lambda = (r' * pho) / a_dot_p;
%   u = u + lambda * pho;
%   r = r - lambda * a;
%   inv_C_times_r = C2 \ (C1 \ r);
%   pho = inv_C_times_r - ((inv_C_times_r' * a) / a_dot_p) * pho;
%   iterp=iterp+1;
% end
% toc
tic
while iterp < m_max
   a = A * pho;
  a_dot_p = a' * pho;
  lambda = (r' * pho) / a_dot_p;
  u = u + lambda * pho;
  r = r - lambda * a;
  inv_C_times_r = C2 \ (C1 \ r);
  pho = inv_C_times_r - ((inv_C_times_r' * a) / a_dot_p) * pho;
  
    tolerancep=norm(r)/norm_b;
    if tolerancep<tol
      break
    end
 iterp=iterp+1;
%  fprintf('n=%g tolerancep=%g\n',iterp,tolerancep)
end
toc
  u;
  iterp;
  

%%
%n=5
%m=4
% qfinal
qexact =p;
qdM = p;
qj = p;
qg = p;
qs = p;
qc = p;
qp = p;
qpnew=p;
directMethod = A\b;

for i=0:m-1
    for j=1:n
        qexact(i+2,j+1)=bpoisson(n*i+j,1);
        qdM(i+2,j+1)=directMethod(n*i+j,1);
        qj(i+2,j+1)=xj(n*i+j,1);
        qg(i+2,j+1)=xg(n*i+j,1);
        qs(i+2,j+1)=xs(n*i+j,1);
        qc(i+2,j+1)=xc(n*i+j,1);
        qp(i+2,j+1)=u(n*i+j,1);
      
    end
end

iterationjacobi=jc
iterationgauss=g
iterationsor=s
iterationc=ic
iterationpre=iterp

%%
%graph
% 
[x2d,y2d]=meshgrid(xO:dx:xL,yO:dy:yL); %for plot graph
contour(x2d,y2d,qexact,'r','ShowText','on')
hold on
contour(x2d,y2d,qp,'b','ShowText','on')
c = sprintf('U(x,y)');
title(c);
ylabel('y');
xlabel('x');
% contour(x2d,y2d,qs,'r','ShowText','on')
% hold on
% contour(x2d,y2d,qj,'b','ShowText','on')
% hold on
% contour(x2d,y2d,qg,'y','ShowText','on')
% hold on
% contour(x2d,y2d,qs,'g','ShowText','on')
% hold on
% contour(x2d,y2d,qc,'m','ShowText','on')
% hold on
% contour(x2d,y2d,qp,'k','ShowText','on')


end
