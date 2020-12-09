//
//  ViewController.m
//  opencv
//
//  Created by weiqing.twq on 2020/12/4.
//

#import <opencv2/opencv.hpp>
#import <opencv2/features2d.hpp>
#import <opencv2/xfeatures2d/nonfree.hpp>
#import <opencv2/calib3d/calib3d.hpp>
#import <opencv2/imgcodecs/ios.h>


#import "ViewController.h"

@interface ViewController ()

@end

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
@implementation ViewController


/// 加载图片同时去除阴影
/// @param name 图片名字
-(Mat)loadImageWithOutShadow:(NSString *)name {
    UIImage *img = [UIImage imageNamed:name];
    Mat ori,gray;
    UIImageToMat(img, ori);
    cvtColor(ori, gray, COLOR_BGR2GRAY);
    //去除图片上的阴影
    std::vector<cv::Mat> rgbChannels(3);
    std::vector<cv::Mat> result_planes;
    std::vector<cv::Mat> result_norm_planes;
    cv::split(gray, rgbChannels);
    const int dilation_size = 3;
    Mat element = getStructuringElement( MORPH_RECT,
                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                           cv::Point( dilation_size, dilation_size ) );
    std::for_each(rgbChannels.begin(), rgbChannels.end(), [&](const Mat &plane) {
        Mat dilated_img;
        cv::dilate(plane, dilated_img, element);
        Mat bg_img;
        medianBlur(dilated_img, bg_img, 21);
        Mat diff_img;
        absdiff(plane, bg_img, diff_img);
        diff_img = 255 - diff_img;
        Mat norm_img;
        normalize(diff_img, norm_img, 0, 255,NORM_MINMAX,CV_8UC1);
        result_planes.push_back(diff_img);
        result_norm_planes.push_back(norm_img);
    });
    Mat result,result_norm;
    merge(result_planes.data(), result_planes.size(), result);
    merge(result_norm_planes.data(), result_norm_planes.size(), result_norm);
    return result;
}

-(void)detectAndComputer:(Mat &)inputImg minHessian:(int)minHessian  descriptor:(Mat &)descriptor keypoints:(vector<KeyPoint> &)keypoints{
    //replace SURF With SIFT
    cv::Ptr<SIFT> detector = SIFT::create(minHessian);
    detector->detect(inputImg, keypoints);
    cv::Ptr<SIFT> extractor = SIFT::create(minHessian);
    extractor->compute(inputImg, keypoints, descriptor);
}


/// 暴力匹配
/// @param goodMatches 返回匹配点
/// @param d1 匹配描述符1
/// @param d2 匹配描述符2
-(void)doBFMatcher:(vector<DMatch> &) goodMatches d1:(Mat &)d1 d2:(Mat &)d2 {
    vector<DMatch> matches;

    BFMatcher matcher = BFMatcher();
    matcher.match(d1, d2, matches);

    // 匹配对筛选
    double min_dist = 1000, max_dist = 0;
    // 找出所有匹配之间的最大值和最小值
    for (int i = 0; i < d1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    // 当描述子之间的匹配不大于2倍的最小距离时，即认为该匹配是一个错误的匹配。
    // 但有时描述子之间的最小距离非常小，可以设置一个经验值作为下限
    for (int i = 0; i < d1.rows; i++)
    {
        if (matches[i].distance <= max(5 * min_dist, 30.0))
            goodMatches.push_back(matches[i]);
    }
}


/// 使用KNN算法match
/// @param goodMatches 返回匹配点
/// @param descriptors1 匹配描述符1
/// @param descriptors2 匹配描述符2
-(void)doKnnMatch:(vector<DMatch> &) goodMatches d1:(Mat &)descriptors1 d2:(Mat &)descriptors2 {
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.8f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            goodMatches.push_back(knn_matches[i][0]);
        }
    }
}

-(void)demo {
    Mat d1,d2;
    vector<KeyPoint> kp1,kp2;
    Mat img1 = [self loadImageWithOutShadow:@"image2.jpg"];
    Mat img2 = [self loadImageWithOutShadow:@"image1.jpg"];
    //二值化
//    threshold(img1, img1, 254, 255, THRESH_BINARY );
    //关键点数量
    int minHessian = 80;
    [self detectAndComputer:img1 minHessian:minHessian descriptor:d1 keypoints:kp1];
    [self detectAndComputer:img2 minHessian:minHessian descriptor:d2 keypoints:kp2];
    //比较
    Mat outImg;
    vector<DMatch> goodMatches;
    [self doBFMatcher:goodMatches d1:d1 d2:d2];
 
    //1.draw keypoint
//    cv::drawKeypoints(img1, kp1, d1);
//    cv::drawKeypoints(img2, kp2, d2);
//    {
//        UIImage* grayImg2 = MatToUIImage(d1);
//        UIImageView *imgView2 = [[UIImageView alloc] initWithImage:grayImg2];
//        [self.view addSubview:imgView2];
//    }
//    {
//        UIImage* grayImg2 = MatToUIImage(d2);
//        UIImageView *imgView2 = [[UIImageView alloc] initWithImage:grayImg2];
//        imgView2.frame =CGRectMake(500, 0, imgView2.frame.size.width, imgView2.frame.size.height);
//        [self.view addSubview:imgView2];
//    }
    
    //2.draw matches
    cv::drawMatches(img1, kp1, img2, kp2, goodMatches, outImg);
    UIImage* outputImg = MatToUIImage(outImg);
    UIImageView *outputImgView = [[UIImageView alloc] initWithImage:outputImg];
    CGFloat width = self.view.frame.size.height / outputImgView.frame.size.height * outputImgView.frame.size.width;
    outputImgView.frame = CGRectMake(200,0,width, self.view.frame.size.height);
    [self.view addSubview:outputImgView];
}

/// 获取角点
-(void)doHarris {
    UIImage *img1 = [UIImage imageNamed:@"church.png"];
    Mat image01,image02,image1,image2;
    UIImageToMat(img1, image01);
    cvtColor(image01, image1, COLOR_BGR2GRAY);

    //提取特征点
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image1, corners, 200, 0.01, 10);
    std::vector<cv::Point2f>::const_iterator it= corners.begin();
    while (it!=corners.end()) {
        cv::circle(image1,*it,3,cv::Scalar(255,255,255),2);
        ++it;
    }
   
    UIImage* grayImg2 = MatToUIImage(image1);
    UIImageView *imgView2 = [[UIImageView alloc] initWithImage:grayImg2];
    [self.view addSubview:imgView2];
}


- (void)viewDidLoad {
    [super viewDidLoad];
    [self demo];
//    [self doHarris];
}

@end
