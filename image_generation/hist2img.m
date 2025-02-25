%function hist2img(histfilename)
    %dir;
    %disp(histfilename);
    tic;
    res = 1.52; %camera resolution microns per pixel

    histfiles = dir('*.hist');
    histfilename = histfiles(1).name;
    fid = fopen(histfilename);
    if fid == -1
        disp(['Error opening hist file  :' histfilename]);
    else
        disp(['Reading histogram file: ' histfilename]);
    end

    [~, filename, ~] = fileparts(histfilename);
    fn_image = [filename '.tif'];

    %fn_image = 'my.tif';
    %fid = fopen('histogram.hist');

    fileflag = fread(fid,11,'*char*1');

    if not(all(fileflag'=='HISTMG V1.0'))
       error('Not a valid histogram file.')
    end

    %disp(['Reading histogram file: ' histfilename]);
    %fprintf('Reading histogram file...\n');

    xsize = fread(fid,1,'int32');
    ysize = fread(fid,1,'int32');
    zsize = fread(fid,1,'int32');

    bitmap = fread(fid,'uint16');
    fclose(fid);

    bitmap = reshape(bitmap,xsize,ysize,zsize);

    maxx = max(max(max(bitmap)));
    bitmap = bitmap*(256/maxx);

    x = 0:res:(xsize-1)*res;
    y = 0:res:(ysize-1)*res;
    z = 0:res:(zsize-1)*res;

    imslice = zeros(xsize,ysize,zsize);

    % Collapse in  plane
    offset = abs(y-mean(y)); % distance from central layer in microns
    %for xy plane offset = abs(z-mean(z));
    int = 1/7;% value of 1/7 from string calib, 0.65 from eye, 2.58 from lens model
    blurmag = offset * int;%.^3; % magnitude of gaussian blur in pixels, technically the standard deviation
    idebug = 0;


    filtsize = round(20*blurmag)+10; % Size of square filter matrix (should be whole image, typically 3*sd is enough)

    finalimage = zeros(xsize,zsize); %create a blank canvas for composite image
    %for xy plane finalimage = zeros(xsize,ysize);

    fprintf('Blurring...\n');

    for i = 1:ysize
        imslice = squeeze(bitmap(:,i,:));

        if blurmag(i) > 0
            % Apply Gaussian blur with imgaussfilt
            blurredimslice = imgaussfilt(imslice, blurmag(i));
            finalimage = finalimage + blurredimslice;
        else
            finalimage = finalimage + imslice;
        end
    end

    %max(finalimage(:))
    finalimage = finalimage.*2./max(finalimage(:));

    %try
    %    imtool(finalimage);
    %catch
    %end

    imwrite(finalimage,fn_image,'tif');
    disp(['Image created: ' fn_image]);

    %disp('Memory used:')
    %s=whos();
    %disp(sum([s.bytes]));
    toc;

%end
