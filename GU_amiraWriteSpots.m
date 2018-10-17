function GU_amiraWriteSpots(filename,xyzCoord, Density, DensityIdx, labelidx, varargin)
% Write an Amira Mesh file with name [<filename>_%04d.am] representing tracks.
% generate the nodes for xyz positions: xyz row(different points), columns(different dimension)

% Gokul Upadhyayula, Oct 2017

ip=inputParser();
ip.CaseSensitive = false;
ip.KeepUnmatched = true;
ip.addRequired('filename', @isstr);
ip.addRequired('xyzCoord', @ismatrix);
ip.addRequired('Density', @isvector);
ip.addRequired('DensityIdx');
ip.addRequired('labelidx');
ip.addParameter('scales', [1,1,1]);

ip.parse(filename,xyzCoord, Density, DensityIdx,labelidx, varargin{:});
p=ip.Results;

s = p.scales;
[pathstr, name, ~] = fileparts(filename);
basename=[pathstr filesep name];

if ~exist(fileparts(filename),'dir')
    mkdir(fileparts(filename));
end

xyzCoord(:,1) = xyzCoord(:,1)*s(1);
xyzCoord(:,2) = xyzCoord(:,2)*s(2);
xyzCoord(:,3) = xyzCoord(:,3)*s(3);
xyzCoord(:,4) = Density;

if ~isempty(labelidx)
    xyzCoord(:,5) = labelidx;
else
    xyzCoord(:,5) = 1:numel(Density);
end

if ~isempty(DensityIdx)
    xyzCoord(:,6) = DensityIdx(:)';
else
    xyzCoord(:,6) = 0;
end

frameFilename=[basename,'.am'];
fid = fopen(frameFilename, 'w');
fprintf(fid,['# PSI Format 1.0\n']);
fprintf(fid,['# column[0] = "x"\n']);
fprintf(fid,['# column[1] = "y"\n']);
fprintf(fid,['# column[2] = "z"\n']);
fprintf(fid,['# column[3] = "Density"\n']);
fprintf(fid,['# column[4] = "Id"\n']);
fprintf(fid,['# column[5] = "DANSynapse"\n']);

fprintf(fid,['# symbol[3] = "D"\n']);
fprintf(fid,['# symbol[5] = "S"\n']);

fprintf(fid,['#type[4] = byte\n']);
fprintf(fid,['#type[5] = byte\n\n']);

fprintf(fid,[num2str(numel(Density)) ' 0 0\n']);
fprintf(fid,['1.00 0.00 0.00\n']);
fprintf(fid,['0.00 1.00 0.00\n']);
fprintf(fid,['0.00 0.00 1.00\n\n']);

dlmwrite(frameFilename, xyzCoord, '-append', 'delimiter',' ','precision', 16);
