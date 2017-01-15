cellSize = 64;
cd('images/');
rawFolder = 'raw/';
archivesFolder = 'archives/';
% searchRawWildcard = fullfile(forawFolder,  '*.jpg');
searchZipWildcard = fullfile(archivesFolder,  '*.zip');
filenames = dir(searchZipWildcard);




% rgbImage = imread('cameraman.tif');

% ca = mat2cell(rgbImage,cellSize * ones(1,size(rgbImage,1)/ cellSize ), cellSize * ones(1,size(rgbImage,2)/ cellSize ),numberOfColorBands);
% plotIndex = 1;
% for c = 1 : size(ca, 2)
% 	for r = 1 : size(ca, 1)
% 		fprintf('c=%d, r=%d\n', c, r);
% 		subplot(rows / cellSize, columns / cellSize,plotIndex);
% 		imshow(ca{r,c});
% 		plotIndex = plotIndex + 1
% 	end
% end
% Enlarge figure to full screen.
% set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

% figure(2)
% imshow(rgbImage);

gaborArray = gaborFilterBank(5,6,39,39);  % Generates the Gabor filter bank
% featuresTable = Table();
featuresCellArray = {};
id = 0;
for f = 1 : size(filenames,1)
    zipfullFileName = fullfile(archivesFolder, filenames(f).name);
    [pathstr,name,ext] = fileparts(zipfullFileName);   
    extractedJPGfullFileName = strcat(name, '.jpg');
    
    
%     system(['unzip ' zipfullFileName ' ' extractedJPGfullFileName ' -d ' outfolder]);
    disp(strcat('Unzipping file # ')); disp(f); disp(' out of total files of '); disp(size(filenames,1));
    system(['unzip ' zipfullFileName ' ' extractedJPGfullFileName]);
%     fullFileName
    % fullFileName = fullfile(folder, baseFileName)
    if ~exist(extractedJPGfullFileName, 'file')
        % Didn't find it there.  Check the search path for it.
        fullFileName = baseFileName; % No path this time.
        if ~exist(extractedJPGfullFileName, 'file')
            % Still didn't find it.  Alert user.
            errorMessage = sprintf('Error: %s does not exist.', extractedJPGfullFileName);
            uiwait(warndlg(errorMessage));
            return;
        end
    end
    
    rgbImage = imread(extractedJPGfullFileName);
    [rows, columns, numberOfColorBands] = size(rgbImage);
    ca = mat2cell(rgbImage,cellSize * ones(1,size(rgbImage,1)/ cellSize ), cellSize * ones(1,size(rgbImage,2)/ cellSize ),numberOfColorBands);
    cellDimX = rows / cellSize;
    cellDimY = columns / cellSize;
    [pathstr,name,ext] = fileparts(extractedJPGfullFileName);   
    for c = 1 : size(ca, 2)
        for r = 1 : size(ca, 1)
%             cellContents = cell2mat(ca{r,c}); % Convert from cell to double.
            featureVector = gaborFeatures(ca{r,c},gaborArray,1,1);
%             size(featureVector);
%             fileNameToWrite = sprintf('%s_cell_%d_%d.jpg', name, r, c);
%             fullFileNameToWrite = fullfile(targetFolder, name);
%             fullFileNameToWrite = fileNameToWrite;
%             imwrite( ca{r, c}, fullFileNameToWrite );
            featuresCellArray = [featuresCellArray; {id, extractedJPGfullFileName, featureVector}];
            id = id + 1;
            
        end
    end
    delete(extractedJPGfullFileName);
    disp(strcat('deleted file...', extractedJPGfullFileName));
end
featuresTable = cell2table(featuresCellArray,'VariableNames',{'ID', 'Filename','Features'});
writetable(featuresTable,'tabledata.dat');
type('tabledata.dat')
disp('Finished!');



% grayImage = reshape(cellContents, [rows, columns]);
% imshow(grayImage, []);


% img = imread('cameraman.tif');
% size(img)

% featureVector = gaborFeatures(img,gaborArray,1,1);   % Extracts Gabor feature vector, 'featureVector', from the image, 'img'.
% size(featureVector)