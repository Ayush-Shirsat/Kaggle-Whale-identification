M = readtable('./all/train.csv');
ma = table2array(M);
imname = [];
for i = 1:length(ma)
    if ma(i,2) ~= "new_whale"
        file = ma(i,1);
        itr = 0;

        I = imread("./all/train/"+ma(i,1));
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(I,filename);
        itr = itr+1;

        I_blur = imgaussfilt(I,2);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(I_blur,filename);
        itr = itr+1;

        img1 = imrotate(I,90);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img1,filename);
        itr = itr+1;

        img2 = imrotate(I,180);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img2,filename);
        itr = itr+1;

        img3 = imrotate(I,270);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img3,filename);
        itr = itr+1;

        img4 = imrotate(I_blur,90);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img4,filename);
        itr = itr+1;

        img5 = imrotate(I_blur,180);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img5,filename);
        itr = itr+1;

        img6 = imrotate(I_blur,270);
        filename = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img6,filename);
        itr = itr+1;
        
    end
end

