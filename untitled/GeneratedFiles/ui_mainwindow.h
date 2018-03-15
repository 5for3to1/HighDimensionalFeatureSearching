/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableView>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QLabel *title;
    QPushButton *chooseFace;
    QLabel *choosenFace;
    QPushButton *search;
    QLabel *result_1;
    QLabel *result_2;
    QLabel *similarity_2;
    QLabel *result_3;
    QLabel *similarity_1;
    QLabel *similarity_3;
    QLabel *result_4;
    QLabel *similarity_4;
    QLabel *result_5;
    QLabel *similarity_5;
    QTableView *myTableView;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(779, 595);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        title = new QLabel(centralWidget);
        title->setObjectName(QStringLiteral("title"));
        title->setGeometry(QRect(270, 30, 191, 31));
        QFont font;
        font.setPointSize(15);
        font.setBold(true);
        font.setItalic(false);
        font.setWeight(75);
        title->setFont(font);
        title->setTextFormat(Qt::AutoText);
        chooseFace = new QPushButton(centralWidget);
        chooseFace->setObjectName(QStringLiteral("chooseFace"));
        chooseFace->setGeometry(QRect(120, 100, 141, 31));
        choosenFace = new QLabel(centralWidget);
        choosenFace->setObjectName(QStringLiteral("choosenFace"));
        choosenFace->setGeometry(QRect(440, 90, 181, 131));
        search = new QPushButton(centralWidget);
        search->setObjectName(QStringLiteral("search"));
        search->setGeometry(QRect(120, 170, 75, 31));
        result_1 = new QLabel(centralWidget);
        result_1->setObjectName(QStringLiteral("result_1"));
        result_1->setGeometry(QRect(60, 260, 61, 61));
        result_1->setFrameShape(QFrame::NoFrame);
        result_1->setLineWidth(1);
        result_2 = new QLabel(centralWidget);
        result_2->setObjectName(QStringLiteral("result_2"));
        result_2->setGeometry(QRect(210, 261, 61, 61));
        similarity_2 = new QLabel(centralWidget);
        similarity_2->setObjectName(QStringLiteral("similarity_2"));
        similarity_2->setGeometry(QRect(210, 350, 54, 12));
        result_3 = new QLabel(centralWidget);
        result_3->setObjectName(QStringLiteral("result_3"));
        result_3->setGeometry(QRect(370, 261, 61, 61));
        similarity_1 = new QLabel(centralWidget);
        similarity_1->setObjectName(QStringLiteral("similarity_1"));
        similarity_1->setGeometry(QRect(60, 350, 54, 12));
        similarity_3 = new QLabel(centralWidget);
        similarity_3->setObjectName(QStringLiteral("similarity_3"));
        similarity_3->setGeometry(QRect(370, 350, 54, 12));
        result_4 = new QLabel(centralWidget);
        result_4->setObjectName(QStringLiteral("result_4"));
        result_4->setGeometry(QRect(523, 261, 61, 61));
        similarity_4 = new QLabel(centralWidget);
        similarity_4->setObjectName(QStringLiteral("similarity_4"));
        similarity_4->setGeometry(QRect(520, 350, 54, 12));
        result_5 = new QLabel(centralWidget);
        result_5->setObjectName(QStringLiteral("result_5"));
        result_5->setGeometry(QRect(663, 261, 61, 61));
        similarity_5 = new QLabel(centralWidget);
        similarity_5->setObjectName(QStringLiteral("similarity_5"));
        similarity_5->setGeometry(QRect(670, 350, 54, 12));
        myTableView = new QTableView(centralWidget);
        myTableView->setObjectName(QStringLiteral("myTableView"));
        myTableView->setGeometry(QRect(40, 370, 681, 131));
        myTableView->setFrameShape(QFrame::Panel);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 779, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\345\244\247\350\247\204\346\250\241\344\272\272\350\204\270\346\243\200\347\264\242\347\263\273\347\273\237", Q_NULLPTR));
        title->setText(QApplication::translate("MainWindow", "\345\244\247\350\247\204\346\250\241\344\272\272\350\204\270\346\243\200\347\264\242\347\263\273\347\273\237", Q_NULLPTR));
        chooseFace->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\345\276\205\346\243\200\347\264\242\347\232\204\344\272\272\350\204\270\345\233\276\347\211\207", Q_NULLPTR));
        choosenFace->setText(QString());
        search->setText(QApplication::translate("MainWindow", "\345\274\200\345\247\213\346\243\200\347\264\242", Q_NULLPTR));
        result_1->setText(QString());
        result_2->setText(QString());
        similarity_2->setText(QString());
        result_3->setText(QString());
        similarity_1->setText(QString());
        similarity_3->setText(QString());
        result_4->setText(QString());
        similarity_4->setText(QString());
        result_5->setText(QString());
        similarity_5->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
