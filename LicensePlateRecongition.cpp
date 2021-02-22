#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <time.h>
#include <sstream>

using namespace std;
using namespace cv;

void detection(Mat frame);
CascadeClassifier plate_classifier;

int main()
{
	plate_classifier.load("haarcascade.xml"); //load cascade file
	VideoCapture vCap(0); //connecting camera
	Mat frame;

	while (vCap.read(frame))
	{
		if (frame.empty())	cout << "Can not connect to camera!" << endl;

		detection(frame);

		if (waitKey(20) == 27) //exit with "esc" button
		{
			break;
		}
	}
	return 0;
}

void detection(Mat frame)
{
	//processing original image for easy recognition
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	Mat element = getStructuringElement(MORPH_RECT, Size(13, 5), Point(-1, -1));
	morphologyEx(frame_gray, frame_gray, MORPH_BLACKHAT, element, Point(-1, -1), 9);

	std::vector<Rect> plates;

	//detect plate with classifier
	plate_classifier.detectMultiScale(frame_gray, plates, 1.1, 15);

	//generate random name for save
	stringstream ss;
	string sa;
	srand(time(NULL));
	sa = to_string(rand());
	ss << sa << ".jpg";

	//draw rectangle around plate
	for (int i = 0;i < plates.size();i++)
	{
		Mat imgcrop = frame(plates[i]);
		rectangle(frame, plates[i], Scalar(0, 0, 255), 5);
		imshow("Capture - Plate ", imgcrop);
		imwrite(ss.str(), imgcrop); //save dedected plate image
	}
	imshow("Capture - Plate detection", frame);
}