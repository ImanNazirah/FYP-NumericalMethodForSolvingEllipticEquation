function varargout = gui1(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui1_OpeningFcn, ...
                   'gui_OutputFcn',  @gui1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function gui1_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

function varargout = gui1_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function gA_Callback(hObject, eventdata, handles)
function gA_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gb_Callback(hObject, eventdata, handles)
function gb_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gclear_Callback(hObject, eventdata, handles)
cla

function gyO_Callback(hObject, eventdata, handles)
function gyO_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gxL_Callback(hObject, eventdata, handles)
function gxL_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gxO_Callback(hObject, eventdata, handles)
function gxO_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gyL_Callback(hObject, eventdata, handles)
function gyL_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gn_Callback(hObject, eventdata, handles)
function gn_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gm_Callback(hObject, eventdata, handles)
function gm_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gf_Callback(hObject, eventdata, handles)
function gf_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gleft_Callback(hObject, eventdata, handles)
function gleft_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gright_Callback(hObject, eventdata, handles)
function gright_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gdown_Callback(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gup_Callback(hObject, eventdata, handles)
function gup_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in gfdm.
function gfdm_Callback(hObject, eventdata, handles)

xO=str2num(get(handles.gxO,'String'));
xL=str2num(get(handles.gxL,'String'));
yO=str2num(get(handles.gyO,'String'));
yL=str2num(get(handles.gyL,'String'));

n=str2num(get(handles.gn,'String'));       %n - mesh for x (in matrices for column,j)
m=str2num(get(handles.gm,'String'));       %m - mesh for y (in matrices for row,i )
dx=(xL-xO)/(n+1)
dy=(yL-yO)/(m+1)


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
Ag=num2str(A);
set(handles.gA,'Max',2,'String',Ag); 
conditionalA=cond(A,Inf)

%%
%input from gui in order to construct matrix B
% fg=get(handles.gf,'String');
% f=str2func(['@(x,y)',fg]);
% bpoisson
fg=get(handles.gf,'String');
leftg=get(handles.gleft,'String');
rightg=get(handles.gright,'String');
downg=get(handles.gdown,'String');
upg=get(handles.gup,'String');
%%
%set up BC's (INSERT EQUATION FOR BOUNDARY EQUATIONS)
x=linspace(xO,xL,n+2);
y=linspace(yO,yL,m+2);

%uleft U(x0,y)
tle=str2func(['@(x,y)',leftg]);       % 0
%uright  U(xL,y) 
tr=str2func(['@(x,y)',rightg]);   % 2*exp(1).^y
%north U(x,yL)  %edit by iman % cth buku burden
tn=str2func(['@(x,y)',upg]); % exp(1)*x  
%usouth U(x,yO)
tso=str2func(['@(x,y)',downg]);     % x


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
f=str2func(['@(x,y)',fg]);
 % Uxx+Uyy=f to insert equation 

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
bg=num2str(b);
set(handles.gb,'Max',2,'String',bg);
%%    
%Jacobi &  Gauss Seidel & SOR
% dxsolve = (xL-xO)/(n*m-1);
% xsolve = xO:dxsolve:xL;
xsolve=zeros(n*m,1);
xj = xsolve;
xg = xsolve;
xs = xsolve;

L =tril(-A,-1);         %untuk hasilkan triangular(figure out why (-A,-1)
D = diag(diag(A));      %untuk hasilkan diagonal A. diag(diag(A)) untuk keluar bentuk
U = triu(-A,1);         %untuk hasilkan triangular(figure out why (-A,1)

Tj = inv(D)*(L+U);       % hampir sama dengan T = inv(D)*(A-D) except -+
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
    xj(:,jc+1)=Tj*xj(:,jc)+Cj;     %%figure out kan mana dapat formula
    %tolerance
    tolerancej = norm(xj(:,jc+1)-xj(:,jc),inf)/norm(xj(:,jc+1),inf);
   
    if tolerancej<delta
        break
    end
     jc=jc+1;              %mesti ada bila buat while
     fprintf('n=%g tolerancej=%g\n',jc,tolerancej)
end
toc
tolerancej;
g=1;
tic 
while g<=N              %N is the iteration
    xg(:,g+1)=Tg*xg(:,g)+Cg;
    %tolerance    
    toleranceg = norm(xg(:,g+1)-xg(:,g),inf)/norm(xg(:,g+1),inf);
    if toleranceg<delta
        break
    end
    g=g+1;              %mesti ada bila buat while
    fprintf('n=%g toleranceg=%g\n',g,toleranceg)
end
toc
s=1;
tic
while s<=N
    xs(:,s+1) = Tw*xs(:,s)+Cw;
    %tolerance
%     tolerances = norm(xs(:,s+1)-xs(:,s),inf)/norm(xs(:,s+1),inf);
    tolerances = norm(xs(:,s+1)-xs(:,s));
    if tolerances < delta
        break
    end
    s=s+1;
    fprintf('n=%g tolerances=%g\n',s,tolerances)
end
toc
xj = xj(:,jc+1);
xg = xg(:,g+1);
xs = xs(:,s+1);
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
xc=x;
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
    fprintf('n=%g tolerancec=%g\n',ic,tolc)
end
toc
xc = xc(:,(ic+1));
ic;
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
 fprintf('n=%g tolerancep=%g\n',iterp,tolerancep)
end
toc
  u;
  iterp;
  
%%
%Solution iteration

gjc = num2str(jc);
gg = num2str(g);
gs = num2str(s);
gic = num2str(ic);
giterp = num2str(iterp);

set(handles.gij,'String',gjc);
set(handles.gig,'String',gg);
set(handles.gis,'String',gs);
set(handles.gic,'String',gic);
set(handles.gip,'String',giterp);

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


%%
%graph
% 
[x2d,y2d]=meshgrid(xO:dx:xL,yO:dy:yL); %for plot graph
axes(handles.graph);
contour(x2d,y2d,qs,'b','ShowText','on')
hold on
contour(x2d,y2d,qj,'b','ShowText','on')
hold on
contour(x2d,y2d,qg,'b','ShowText','on')
hold on
contour(x2d,y2d,qs,'b','ShowText','on')
hold on
contour(x2d,y2d,qc,'b','ShowText','on')
hold on
contour(x2d,y2d,qp,'b','ShowText','on')
xlswrite('myexampleguifdm.xlsx',[xj,xg,xs,xc,u]);


% 
% %subplot(2,3,1)
% %pcolor(x2d,y2d,qdM)
% hold on
% contour(x2d,y2d,qdM)%,'r','ShowText','on')
% %subplot(2,3,2)
% pcolor(x2d,y2d,qj)
%  hold on
 


% --- Executes on button press in gcm.
function gcm_Callback(hObject, eventdata, handles)

xO=str2num(get(handles.gxO,'String'));
xL=str2num(get(handles.gxL,'String'));
yO=str2num(get(handles.gyO,'String'));
yL=str2num(get(handles.gyL,'String'));

n=str2num(get(handles.gn,'String'));       %n - mesh for x (in matrices for column,j)
m=str2num(get(handles.gm,'String'));       %m - mesh for y (in matrices for row,i )
dx=(xL-xO)/(n+1);
dy=(yL-yO)/(m+1);
%%
%Matrix A
for i=1:n*m
    A(i,i)=-2*(1/dx^2+1/dy^2);
end
for i=1:n
end
for i=1:n-1 
    %to construct lower and upper diagonal
    for j=1:m
    A(i+(j-1)*n, i+(j-1)*n+1)= (1/dx^2-2/dy^2); %upper diagonal
    A(i+(j-1)*n+1, i+(j-1)*n)= 1/dx^2;   %lower diagonal
    end
end
%to construct different Block for digonal 
for i=1:n
    for j=1:m-1
        A(i+(j-1)*n,i+j*n)= -2/dx^2+1/dy^2;     %block upper(diagonal)
       A(i+j*n,i+(j-1)*n)= 1/dy^2;            %block lower(diagonal)
    end  
end
for i=1:n-1 
    %to construct lower and upper diagonal for different block
    for j=1:m-1
    A(i+(j-1)*n, i+(j-1)*n+n+1)= 1/dx^2+1/dy^2;  %purple    
    A(i+(j-1)*n+1, i+(j-1)*n+n)= 1/dx^2;%turaqouise
  A(i+(j-1)*n+n, i+(j-1)*n+1)=1/dy^2;  %blue
    end
   
end

Ag=num2str(A);
set(handles.gA,'Max',2,'String',Ag)
conditionalA=cond(A,Inf)


%%
%input from gui in order to construct matrix B
% fg=get(handles.gf,'String');
% f=str2func(['@(x,y)',fg]);
% bpoisson
fg=get(handles.gf,'String');
leftg=get(handles.gleft,'String');
rightg=get(handles.gright,'String');
downg=get(handles.gdown,'String');
upg=get(handles.gup,'String');
%%
%set up BC's (INSERT EQUATION FOR BOUNDARY EQUATIONS)
x=linspace(xO,xL,n+2)
y=linspace(yO,yL,m+2)

%uleft U(x0,y)
tle=str2func(['@(x,y)',leftg]);       % 0
%uright  U(xL,y) 
tr=str2func(['@(x,y)',rightg]);   % 2*exp(1).^y
%north U(x,yL)  %edit by iman % cth buku burden
tn=str2func(['@(x,y)',upg]); % exp(1)*x  
%usouth U(x,yO)
tso=str2func(['@(x,y)',downg]);     % x


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
 %MATRIX B
 b = zeros(n*m,1);

 
 for j=1:n-2
     b(j+1,1)=(1/dy^2)*(p(1,j+2)+p(1,j+3));	%dah
     b(m*n-j,1)=(1/dx^2)*(p(m+2,n+2-j)+p(m+2,n-j))-(2/dx^2)*p(m+2,n+1-j)+(1/dy^2)*(p(m+2,n+1-j)+p(m+2,n+2-j));   %dah
     
     for k=1:m-2  
     b(k*n+1,1)=(1/dx^2)*(p(k+2,1)+p(k+3,1));
     b(k*n+n,1)=(1/dx^2)*(p(k+2,n+2)+p(k+3,n+2))+(1/dy^2)*(p(k+1,n+2)+p(k+3,n+2))-(2/dy^2)*(p(k+2,n+2)); %dah
     end  
 end
  b(1,1)=(1/dx^2)*(p(2,1)+p(3,1))+(1/dy^2)*(p(1,2)+p(1,3)); %dah
  b(n,1)=(1/dx^2)*(p(2,n+2)+p(3,n+2))+(1/dy^2)*(p(1,n+1)+p(3,n+2)+p(1,n+2))-(2/dy^2)*p(2,n+2); %dah
  b(m*n-(n-1),1)=(1/dx^2)*(p(m+1,1)+p(m+2,3)+p(m+2,1))-(2/dx^2)*p(m+2,2)+(1/dy^2)*(p(m+2,2)+p(m+2,3)); %dah
  b(m*n,1)=(1/dx^2)*(p(m+1,n+2)+p(m+2,n+2)+p(m+2,n))-(2/dx^2)*p(m+2,n+1)+(1/dy^2)*(p(m+2,n+1)+p(m+2,n+2)+p(m,n+2))-(2/dy^2)*p(m+1,n+2); %dah
  
%%
%Matrix bpoisson
          % Uxx+Uyy=f to insert equation 
f=str2func(['@(x,y)',fg]); % exp(1)*x  

bpoisson= zeros(n*m,1);
for k=n
    for i=1:m
        for j=1:n
        bpoisson(j+((i-1)*k),1)=f(x(1,j+1),y(1,i+1));%4
%         x(1,j+1)*exp(1)^ynew(i+1,1);%4*x(1,j+1)+3*ynew(i+1,1); %insert equation for poisson
        end
    end
end
bpoisson;
b=bpoisson-b;
bg=num2str(b);
set(handles.gb,'Max',2,'String',bg);

%%    
%Jacobi &  Gauss Seidel & SOR
% dxsolve = (xL-xO)/(n*m-1);
% xsolve = xO:dxsolve:xL;
xsolve=zeros(n*m,1);
xj = xsolve;
xg = xsolve;
xs = xsolve;

L =tril(-A,-1);         %untuk hasilkan triangular(figure out why (-A,-1)
D = diag(diag(A));      %untuk hasilkan diagonal A. diag(diag(A)) untuk keluar bentuk
U = triu(-A,1);         %untuk hasilkan triangular(figure out why (-A,1)

Tj = inv(D)*(L+U);       % hampir sama dengan T = inv(D)*(A-D) except -+
Cj = inv(D)*b;
Tg = inv(D-L)*(U);
Cg = inv(D-L)*b;

Ts = inv(D)*(L+U);
rho_Ts = max(abs(eig(Ts)));
 w = 2/(1 + sqrt(1-rho_Ts^2));
%w=1.25
% w=0.25

Tw = inv(D-w*L)*((1-w)*D+w*U);
Cw = w*(inv(D-w*L))*b;
delta = 1e-5;          %mesti ada bila buat tolerance
N=10000000000000;
jc=1;
tic%mesti ada bila buat while
while jc<=N              %N is the iteration
    xj(:,jc+1)=Tj*xj(:,jc)+Cj;     %%figure out kan mana dapat formula
    %tolerance
    tolerancej = norm(xj(:,jc+1)-xj(:,jc),inf)/norm(xj(:,jc+1),inf);
   
    if tolerancej<delta
        break
    end
     jc=jc+1;              %mesti ada bila buat while
    fprintf('n=%g tolerancej=%g\n',jc,tolerancej) 
end
toc
tolerancej;
g=1;
tic
while g<=N              %N is the iteration
    xg(:,g+1)=Tg*xg(:,g)+Cg;
    %tolerance    
    toleranceg = norm(xg(:,g+1)-xg(:,g),inf)/norm(xg(:,g+1),inf);
    if toleranceg<delta
        break
    end
    g=g+1;              %mesti ada bila buat while  
    fprintf('n=%g toleranceg=%g\n',g,toleranceg)
end
toc
s=1;
tic
while s<=N
    xs(:,s+1) = Tw*xs(:,s)+Cw;
    %tolerance
   tolerances = norm(xs(:,s+1)-xs(:,s),inf)/norm(xs(:,s+1),inf);
    %tolerances = norm(xs(:,s+1)-xs(:,s));
    if tolerances < delta
        break
    end
    s=s+1;
    fprintf('n=%g tolerances=%g\n',s,tolerances)
end
toc
xj = xj(:,jc+1);
xg = xg(:,g+1);
xs = xs(:,s+1);

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
 fprintf('n=%g tolerancep=%g\n',iterp,tolerancep)
end
toc
  u;
  iterp;
  
%%
%Solution iteration
gjc = num2str(jc);
gg = num2str(g);
gs = num2str(s);
gic = num2str(0);
giterp = num2str(iterp);

set(handles.gij,'String',gjc);
set(handles.gig,'String',gg);
set(handles.gis,'String',gs);
set(handles.gic,'String',gic);
set(handles.gip,'String',giterp);

%%
%n=5
%m=4
% qfinal
qexact = p;
qdM = p;
qj = p;
qg = p;
qs = p;
qp = p;
directMethod = A\b;

for i=0:m-1
    for j=1:n
        %qexact(i+2,j+1)=ex(n*i+j,1);
        qdM(i+2,j+1)=directMethod(n*i+j,1);
        qj(i+2,j+1)=xj(n*i+j,1);
        qg(i+2,j+1)=xg(n*i+j,1);
        qs(i+2,j+1)=xs(n*i+j,1);
        qp(i+2,j+1)=u(n*i+j,1);
    end
end

%%
%graph
% 
[x2d,y2d]=meshgrid(xO:dx:xL,yO:dy:yL); %for plot graph
axes(handles.graph);
contour(x2d,y2d,qs,'b','ShowText','on')
hold on
contour(x2d,y2d,qj,'y','ShowText','on')
hold on
contour(x2d,y2d,qg,'g','ShowText','on')
hold on
contour(x2d,y2d,qs,'k','ShowText','on')
hold on
contour(x2d,y2d,qp,'m','ShowText','on')
xlswrite('myexampleguicrank.xlsx',[xj,xg,xs,u]);


function gij_Callback(hObject, eventdata, handles)
function gij_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gig_Callback(hObject, eventdata, handles)
function gig_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gic_Callback(hObject, eventdata, handles)
function gic_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gip_Callback(hObject, eventdata, handles)
function gip_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function gis_Callback(hObject, eventdata, handles)
function gis_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
