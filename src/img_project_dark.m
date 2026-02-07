clc; clear; close all;
py.importlib.import_module('tensorflow')
%% ======================= Load Night Image ==========================
img = imread('dark10.png');   % <-- change this to your input image
figure, imshow(img), title('Original Night Image');

% Convert to grayscale (some methods operate on grayscale)
gray = rgb2gray(img);

%% ======================= 1. Histogram Equalization =================
heq = histeq(gray);
figure, imshow(heq), title('Histogram Equalized');

% Compute brightness (0 = black, 1 = white)
brightness = mean(gray(:)) / 255;

% Automatically set ClipLimit based on brightness
if brightness < 0.25
    clip = 0.04;  
elseif brightness < 0.45
    clip = 0.02;
else
    clip = 0.008;  
end

% Auto tile size: use smaller tiles for very dark images
if brightness < 0.25
    tiles = [4 4];
else
    tiles = [8 8];
end

% Apply adaptive CLAHE
clahe = adapthisteq(gray, ...
                    'ClipLimit', clip, ...
                    'NumTiles', tiles, ...
                    'Distribution', 'rayleigh');

figure, imshow(clahe), title(['Auto-CLAHE (ClipLimit=', num2str(clip), ')']);


%% ======================= 3. Gamma Correction ========================
gamma = 0.4;   % <--- can be tuned (0.3–0.6 good for night images)
gamma_corrected = imadjust(gray, [], [], gamma);
figure, imshow(gamma_corrected), title(['Gamma Corrected (γ=', num2str(gamma), ')']);

%% ======================= 4. Median Filter Denoising =================
median_filtered = medfilt2(gray, [3 3]);
figure, imshow(median_filtered), title('Median Filtered');

%% ======================= 5. Bilateral Filtering (Edge-Preserving) ===
bilateral = imbilatfilt(gray, 20, 10);
figure, imshow(bilateral), title('Bilateral Filter (Denoised + Sharp)');

%% ======================= 6. Sharpening ==============================
sharpened = imsharpen(gray, 'Radius',2,'Amount',1.5);
figure, imshow(sharpened), title('Sharpened Image');

%% ======================= 7. Multi-Scale Retinex (MSR) ================
% MSR enhances illumination and preserves details
gray_double = im2double(gray);

scales = [15 80 250];
weights = [1 1 1] / 3;
MSR = zeros(size(gray_double));

for k = 1:length(scales)
    sigma = scales(k);
    blur = imgaussfilt(gray_double, sigma);
    MSR = MSR + weights(k) * log(1 + gray_double ./ blur);
end

% Normalize MSR to [0, 1]
MSR = mat2gray(MSR);

figure, imshow(MSR), title('Multi-Scale Retinex (MSR)');

%% ======================= 8. Color Restoration (optional) ============
% Apply MSR to each channel for color retinex
img_double = im2double(img);
MSR_color = img_double;

for ch = 1:3
    I = img_double(:,:,ch);
    MSR_ch = zeros(size(I));

    for k = 1:length(scales)
        blur = imgaussfilt(I, scales(k));
        MSR_ch = MSR_ch + weights(k)*log(1 + I./blur);
    end

    MSR_color(:,:,ch) = mat2gray(MSR_ch);
end

figure, imshow(MSR_color), title('Color Retinex Enhancement');

%% ======================= 9. FINAL NIGHT VISION PIPELINE =============
% Combine the best enhancements to create a professional night-vision output

% Step-by-step fusion:
fused = im2double(gray);
fused = 0.4*fused + 0.3*im2double(clahe) + 0.3*MSR;    % weighted fusion
fused = imsharpen(fused, 'Radius',1.2, 'Amount',1.0);  % enhance edges
fused = mat2gray(fused);                               % normalize

figure, imshow(fused), title('FINAL Night Vision Enhanced Output');

%% ======================= 10. Quantitative Evaluation =================
% Entropy and Contrast metrics

% Convert images to uint8 if needed
orig_uint8   = im2uint8(gray);
clahe_uint8  = im2uint8(clahe);
msr_uint8    = im2uint8(MSR);
final_uint8  = im2uint8(fused);

%% -------- Entropy Calculation --------
entropy_original = entropy(orig_uint8);
entropy_clahe    = entropy(clahe_uint8);
entropy_msr      = entropy(msr_uint8);
entropy_final    = entropy(final_uint8);

%% -------- Contrast Calculation --------
% Contrast defined as standard deviation of intensity values
contrast_original = std(double(orig_uint8(:)));
contrast_clahe    = std(double(clahe_uint8(:)));
contrast_msr      = std(double(msr_uint8(:)));
contrast_final    = std(double(final_uint8(:)));

%% -------- Display Results --------
fprintf('\n===== Entropy Results =====\n');
fprintf('Original Image   : %.4f\n', entropy_original);
fprintf('CLAHE Image      : %.4f\n', entropy_clahe);
fprintf('MSR Image        : %.4f\n', entropy_msr);
fprintf('Final Enhanced   : %.4f\n', entropy_final);

fprintf('\n===== Contrast Results =====\n');
fprintf('Original Image   : %.4f\n', contrast_original);
fprintf('CLAHE Image      : %.4f\n', contrast_clahe);
fprintf('MSR Image        : %.4f\n', contrast_msr);
fprintf('Final Enhanced   : %.4f\n', contrast_final);

%% ======================= SAVE OUTPUT ================================
imwrite(fused, 'night_vision_output.jpg');
disp('Night Vision Enhancement Completed');


