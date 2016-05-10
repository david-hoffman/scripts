%clear all;
%close all;
addpath('D:\Matlab data\Matlab_calibration\Functions');

%================ Initial Parameters ========================
N_sampling = 256;
NA = 1.1;
dx = 0.114;
%dz = 0.1;
%lamd = 0.605;
index = 1.33;
%DM_FF = 13.2/13.5;
DM_FF = 1;
%============================================================
C_path = pwd;
Previous_acts = importdata([C_path '\Acts\ALP_previous.txt']);

%================= Load PSF =================================
[FileName,PathName,Filterindex] = uigetfile('*.tif','MultiSelect', 'off');
PSF_measured = (loadtiff([PathName char(FileName)]));
%PSF_measured = loadtiff([C_path '\Images\PR_Stack.tif']);


[nx, ny, nz] = size(PSF_measured);
[C,I] = max(PSF_measured(:));
[PSF_center(1), PSF_center(2), PSF_center(3)] = ind2sub(size(PSF_measured), I);
%[nx,ny,nz] = [96,96,100];
%PSF_center = [60, 63,10];
%noise_level = 100;
%noise_variation = 10;
PSF_offset = 150;
%PSF_max = 255;
PSF = zeros(N_sampling,N_sampling,nz);%noise_level*ones(N_sampling,N_sampling,nz)+noise_variation*rand(N_sampling,N_sampling,nz);
PSF((N_sampling/2-PSF_center(1)+1):(N_sampling/2+nx-PSF_center(1)),(N_sampling/2-PSF_center(2)+1):(N_sampling/2+ny-PSF_center(2)),:) = PSF_measured;
PSF = sqrt(PSF);
PSF = (PSF-sqrt(PSF_offset)).*((PSF-sqrt(PSF_offset))>0);

%============================================================


%================= Define Pupil Plane =======================
dk=2*pi/N_sampling/dx;
kx=[-(N_sampling-1)/2:1:(N_sampling-1)/2]*dk;
ky=kx;
[kxx, kyy]=meshgrid(kx,ky);
kr=sqrt(kxx.^2+kyy.^2);
max(max(kr))

pupil_mask=(kr<NA*(2*pi/lamd));
%pupil_intensity = abs(pupil_mask);
%pupil_phase = angle(pupil_mask);
pupil_field = double(pupil_mask);
%=============================================================
Defocus = zeros(N_sampling,N_sampling,nz);
for j = 1:nz
        Defocus(:,:,j) = exp(1i*sqrt(((2*index*pi/lamd)^2 - kr.^2).*((2*index*pi/lamd)^2 > kr.^2))*dz*(j-PSF_center(3)));
end

for i = 1:25
    %========== Generate Vertual PSF from Pupil Function =========
    %PSFA = zeros(N_sampling,N_sampling,nz); 
    %for j = 1:nz
    %    Defocus = exp(1i*sqrt(((2*pi/lamd)^2 - kr.^2).*((2*pi/lamd)^2 > kr.^2))*dz*(j-PSF_center(3)));
    %    PSFA(:,:,j) = fftshift(fft2(ifftshift(pupil_field.*Defocus)));
    %end
    
    %=============================================================
    
    
    %======== Replace Amplitude of PSFA to Measured PSF ==========
    %PSFA = PSF.*exp(1i*angle(PSFA));
    %PSFA = PSFA/max(max(max(abs(PSFA))))*100;
    %write3Dtiff(uint16(abs(PSFA)), 'TEST_PSF.tif');
    
    %==============================================================
    
    %======== Calculate New Pupil Function ========================
    %pupil_field = 0*pupil_field;
    %for j = 1:nz
    %    Defocus = exp(1i*sqrt(((2*pi/lamd)^2 - kr.^2).*((2*pi/lamd)^2 > kr.^2))*dz*(j-PSF_center(3)));
    %    pupil_field = pupil_field + ifftshift(ifft2(fftshift(PSFA(:,:,j))))./Defocus;
    %end
    %pupil_field = pupil_field/nz;
    %=============================================================
    
    %=========== Add Constrains to Pupil Filed====================
    %pupil_amp = abs(pupil_field);
    %pupil_amp = ImageFilter(abs(pupil_field), 'Gaussian' ,N_sampling/100);
    %pupil_phase = ImageFilter(angle(pupil_field), 'Gaussian' ,N_sampling/100);
    %pupil_field = (pupil_amp.*exp(1i*pupil_phase)).*pupil_mask;
    %pupil_field = pupil_field.*pupil_mask;
    %==============================================================
    i
    
    %=========== Conbine Loops Above ==============================
    PSFA = zeros(N_sampling,N_sampling,nz); 
    temp = zeros(N_sampling,N_sampling);
    for j = 1:nz
        %Defocus = exp(1i*sqrt(((2*pi/lamd)^2 - kr.^2).*((2*pi/lamd)^2 > kr.^2))*dz*(j-PSF_center(3)));
        PSFA(:,:,j) = PSF(:,:,j).*exp(1i*angle(fftshift(fft2(ifftshift(pupil_field.*Defocus(:,:,j))))));
        temp = temp + ifftshift(ifft2(fftshift(PSFA(:,:,j))))./Defocus(:,:,j);
    end
    pupil_field = (temp/nz).*pupil_mask;
    pupil_field = pupil_mask.*exp(1i*angle(pupil_field));

end
figure;
%pupil_phase = (angle(pupil_field.*pupil_mask));
pupil_phase = GoldsteinUnwrap2D_r1_TL(pupil_field,pupil_mask);
pupil_phase(isnan(pupil_phase)) = 0;
pupil_displacement = (pupil_phase/2/pi*lamd)';
pupil_displacement = flipdim(pupil_displacement,1);
pupil_displacement = flipdim(pupil_displacement,2);
%surf(pupil_displacement, 'EdgeColor', 'none'); view(2);caxis([-1*max(max(abs(pupil_displacement))), max(max(abs(pupil_displacement)))]);
imagesc(pupil_displacement), colormap(jet), colorbar, axis square, axis off, title('Displacement');caxis([-0.4, 0.4]);caxis([-1*max(max(abs(pupil_displacement))), max(max(abs(pupil_displacement)))]);

x_pupil = kxx(:)/(NA*(2*pi/lamd))*DM_FF; y_pupil = kyy(:)/(NA*(2*pi/lamd))*DM_FF;
[theta_pupil,r_pupil] = cart2pol(x_pupil,y_pupil); 
is_in_circle = ( r_pupil <= 1 );
%pupil_phase(~is_in_circle) = 0;

N = []; M = [];
for n = 0:9
N = [N n*ones(1,n+1)];
M = [M -n:2:n];
end 


Z = zernfun(N,M,r_pupil(is_in_circle),theta_pupil(is_in_circle));
PR_Zcoeficients = Z\pupil_displacement(is_in_circle);
figure;
bar(PR_Zcoeficients);
PR_Zcoeficients(1:3)=0;
%PR_Zcoeficients(5) = 0;
Reconstructed = 0*pupil_displacement;
Reconstructed(is_in_circle) = Z*PR_Zcoeficients;
%Z_modes = zeros(N_sampling,N_sampling,size(Z,2));
%Temp = 0*pupil_phase;
%for n = 1:size(Z,2)
%    Temp(is_in_circle) = Z(:,n);
%    Z_modes(:,:,n) = Temp;
%end
%figure;
%surf(pupil_phase, 'EdgeColor', 'none'); caxis([min(Data(is_in_circle)),max(Data(is_in_circle)) ]); view(2);
figure;
imagesc(Reconstructed), colormap(jet), colorbar, axis square, axis off, title('Reconstructed');caxis([-1*max(max(abs(pupil_displacement))), max(max(abs(pupil_displacement)))]);
%surf(Reconstructed, 'EdgeColor', 'none'); caxis([-1*max(max(abs(pupil_displacement))), max(max(abs(pupil_displacement)))]); view(2);
%save('PR_Zernike_coeficients.mat','PR_Zcoeficients');

load([C_path '\Mats\Zernike_mode_act_basis_6all_55modes.mat']);

acts = Previous_acts -1 * coeficients*PR_Zcoeficients; % * (0.515/(2*pi));

save([C_path '\Acts\ALP_flat.txt'],'acts','-ascii')
save([C_path '\Acts\ALP_previous.txt'],'acts','-ascii')