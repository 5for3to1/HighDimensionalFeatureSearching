#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPalette>
#include<string>

using namespace std;


//存储最终选择出的相似度与图片路径
struct PathSimilarity
{
	string path;
	float similarity;
};

#define face_num 5											//显示人脸的数量


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:

    void on_search_clicked();

    void on_chooseFace_clicked();

private:
    Ui::MainWindow *ui;

	QString choosenFileName;							//选中的人脸图片文件名
	int face_index;										//选中的人脸图片在test_face中的索引号

	string ROOT_PATH = "G:/MyQt/";						//人脸图片和工程所在的路径
	PathSimilarity * pathSimilarity;
};

#endif // MAINWINDOW_H


extern "C"
void red_face_lib();
extern "C"
void read_face_test(int index );
extern "C"
PathSimilarity * calc();
extern "C"
void revoke();
/*
extern "C"
void Topk(int index);
*/