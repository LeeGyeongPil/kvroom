# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

kvroom 은 브라우저에서만 동작하는 정적 앱이며 토큰이나 키를 보내지 않습니다(Hugging Face Hub 공개 config 요청만 하고, 끌 수도 있습니다). 개발 서버(`npm run dev`)는 내 컴퓨터(localhost)에서만 접속되도록 열립니다. / kvroom is a browser-only static app and sends no tokens or keys (it only requests public configs from the Hugging Face Hub, and that can be turned off). The development server (`npm run dev`) listens on localhost only.
