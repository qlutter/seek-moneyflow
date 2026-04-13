# PowerFlow 60m Scanner

미국 주식 장 중 30분마다 자동 스캔 → 텔레그램 알림

## 파일 구조
```
powerflow-repo/
├── .github/
│   └── workflows/
│       └── powerflow.yml     # GitHub Actions 스케줄러
├── powerflow_scan.py          # 메인 스캔 코드
├── requirements.txt           # 라이브러리 목록
└── README.md
```

## 설정 방법

### 1. Secrets 등록
GitHub 레포 → Settings → Secrets and variables → Actions

| 이름 | 설명 |
|------|------|
| `TELEGRAM_TOKEN` | @BotFather에서 받은 봇 토큰 |
| `CHAT_ID` | @userinfobot에서 확인한 ID |

### 2. 실행 시간
- 미국 장 중 (ET 9:30 AM ~ 4:00 PM) 30분마다 자동 실행
- Actions 탭에서 수동 실행도 가능

## 티커 수정
`powerflow_scan.py` 하단의 `tickers` 리스트를 수정하세요.


## ticker.txt 사용법

- 티커 목록은 코드 안에 하드코딩하지 않고 `ticker.txt`에서 읽습니다.
- 기본 경로는 저장소 루트의 `ticker.txt`입니다.
- 한 줄에 하나씩 넣는 방식을 권장합니다.
- 빈 줄, `#`로 시작하는 줄은 무시됩니다.
- 쉼표(`,`)나 공백으로 여러 티커를 한 줄에 적어도 됩니다.
- 다른 파일을 쓰고 싶으면 환경변수 `TICKER_FILE`로 경로를 지정할 수 있습니다.
