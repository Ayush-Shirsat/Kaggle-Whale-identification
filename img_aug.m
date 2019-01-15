
M = readtable('./all/train.csv');
ma = table2array(M);
train_label = [];       
for i = 1:length(ma)
    if ma(i,2) ~= "new_whale"
       
        file = ma(i,1);
        itr = 0;

        I = imread("./all/train/"+ma(i,1));
        f1 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(I,f1);
        itr = itr+1;

        I_blur = imgaussfilt(I,2);
        f2 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(I_blur,f2);
        itr = itr+1;

        img1 = imrotate(I,90);
        f3 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img1,f3);
        itr = itr+1;

        img2 = imrotate(I,180);
        f4 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img2,f4);
        itr = itr+1;

        img3 = imrotate(I,270);
        f5 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img3,f5);
        itr = itr+1;

        img4 = imrotate(I_blur,90);
        f6 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img4,f6);
        itr = itr+1;

        img5 = imrotate(I_blur,180);
        f7 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img5,f7);
        itr = itr+1;

        img6 = imrotate(I_blur,270);
        f8 = sprintf("./newtest/"+file+"_%03d.jpg", itr);
        imwrite(img6,f8);
        itr = itr+1;
        
        fnames = [f1,f2,f3,f4,f5,f6,f7,f8];
        label = ma(i,2);
        for k = 1:length(fnames)
            train_label = [train_label;fnames(k),label];
        end
           
    end
end
