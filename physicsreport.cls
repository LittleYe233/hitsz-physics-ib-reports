\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{physicsreport}[LaTeX template for HITsz Physics IB reports ver. 1.0.0]

%%
%% Class options
%%

% Define user-side options
\RequirePackage{ifthen}
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
    family=physicsreport,
    prefix=physicsreport@opt@
}

\DeclareStringOption[data]{signature}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexart}}

\ProcessKeyvalOptions*
\ifthenelse{\NOT\(\equal{\physicsreport@opt@signature}{data}\OR\equal{\physicsreport@opt@signature}{preparation}\)}{%
    \ClassError{physicsreport}{%
        The option ``signature'' receives an illegal value}{%
        Please check the documentation of this document class}
}

%%
%% Loaded document class and packages
%%

\LoadClass[linespread=1.5, fontset=custom]{ctexart}

\RequirePackage[left=1.25in, right=1.25in, bottom=1in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{tabularx}
% Provides `/Xcline'.
\RequirePackage{makecell}
\RequirePackage{datetime}
\RequirePackage{titlesec}
% Removes vertical gaps in lists
\RequirePackage{enumitem}

%%
%% Pre user settings
%%

\newcommand{\classno}[1]{\def\theclassno{#1}}
\newcommand{\stuno}[1]{\def\thestuno{#1}}
\newcommand{\groupno}[1]{\def\thegroupno{#1}}
\newcommand{\stuname}[1]{\def\thestuname{#1}}
\newcommand{\expdate}[1]{\def\theexpdate{#1}}
\newcommand{\expname}[1]{\def\theexpname{#1}}

\newdateformat{expdatefmt}{\THEYEAR-\THEMONTH-\THEDAY}

% See: https://tex.stackexchange.com/a/281864/290833
\newcommand{\doublerule}[1][.4pt]{%
    \noindent
    \makebox[0pt][l]{\rule[.7ex]{\linewidth}{#1}}%
    \rule[0ex]{\linewidth}{#1}}

%%
%% Package settings
%%

\ctexset{section={
      format=\normalsize\bfseries\raggedright,
      number=\chinese{section},
      name={,、},
      aftername=
     }, subsection={
            format=\normalsize\bfseries\raggedright,
            number=\arabic{subsection},
            name={,.}
        }}

\fancyhead[L]{{\large\kaishu 大学物理实验报告}}
\fancyhead[R]{{\large\kaishu 哈尔滨工业大学(深圳)}}
\fancyfoot[C]{}
\fancyfoot[R]{{\small \thepage}}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{.9pt}
% Resolve fancyhdr head height too small
% See: https://tex.stackexchange.com/a/327298
\setlength{\headheight}{13.98427pt}
\setlength{\topmargin}{-.35in}
\setlength{\headsep}{.2in}

% Remove gaps before and after the titles.
\titlespacing*{\section}{0pt}{0ex}{0ex}
\titlespacing*{\subsection}{0pt}{0ex}{0ex}
\titlespacing*{\subsubsection}{0pt}{0ex}{0ex}

\newcolumntype{Y}{>{\centering\arraybackslash}X}

\setlist[enumerate]{noitemsep}

%%
%% Document body stuff
%%

% Personal information and experiment title on the first page
\renewcommand{\maketitle}{%
    \begin{table*}[h]
        \setlength\tabcolsep{0pt}
        \begin{tabularx}{\textwidth}{XXXX}
            {
                \begin{tabularx}{\linewidth}{cY}
                    \textbf{班号} & \theclassno \\\Xcline{2-2}{1pt}
                \end{tabularx}
            } & {
                    \begin{tabularx}{\linewidth}{cY}
                        \textbf{学号} & \thestuno \\\Xcline{2-2}{1pt}
                    \end{tabularx}
            } & {
                    \begin{tabularx}{\linewidth}{cY}
                        \textbf{姓名} & \thestuname \\\Xcline{2-2}{1pt}
                    \end{tabularx}
            } & {
                    \begin{tabularx}{\linewidth}{cY}
                        \textbf{教师签字} & \\\Xcline{2-2}{1pt}
                    \end{tabularx}
            }     \\
            % Add to the vertical gap between the two lines.
            % We do not specify `/arraystretch' because it widens the vertical
            % height of cells so that the horizontal lines (`\cline') are too away
            % from the text.
            \\[-8pt]
            {
            \begin{tabularx}{\linewidth}{cY}
                \textbf{实验日期} & \theexpdate \\\Xcline{2-2}{1pt}
            \end{tabularx}
            } & {
                    \begin{tabularx}{\linewidth}{cY}
                        \textbf{组号} & \thegroupno \\\Xcline{2-2}{1pt}
                    \end{tabularx}
            } & {
                    \begin{tabularx}{\linewidth}{cY}
                        \textbf{预习成绩} & \\\Xcline{2-2}{1pt}
                    \end{tabularx}
            } & {
                    \begin{tabularx}{\linewidth}{cY}
                        \textbf{总成绩} & \\\Xcline{2-2}{1pt}
                    \end{tabularx}
                }
        \end{tabularx}
    \end{table*}

    \doublerule[.6pt]

    % Begin the experiment title and preparation part.
    \vspace*{10pt}
    \noindent{\large{\centering\textbf{ 实验名称\quad\theexpname}\par}}
    \vspace*{6pt}
}

% Teacher signature
\newcommand{\physicsreport@body@signature@body}{%
    \renewcommand{\arraystretch}{1.8}
    \vspace*{\fill}
    \hfill
    \begin{tabularx}{.35\textwidth}[b]{@{}|Y|Y|@{}} \hline
        \fontsize{14}{16}\selectfont \textbf{教师} & \fontsize{14}{16}\selectfont \textbf{姓名} \\\hline
        \fontsize{16}{18}\selectfont \textbf{签字} &                                          \\\hline
    \end{tabularx}
    \renewcommand{\arraystretch}{1}
}
\newcommand{\physicsreport@body@signature}[1]{%
    \ifthenelse{\equal{\physicsreport@opt@signature}{#1}}{\physicsreport@body@signature@body}{}
}
