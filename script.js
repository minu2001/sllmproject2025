function generateText() {
    const inputText = document.getElementById('userInput').value;
  
    $.ajax({
      url: 'http://localhost:8000/v1/chat/completions',
      type: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        model: "openchat-3.5-1210",
        messages: [{ role: "user", content: inputText }]
      }),
      success: function(response) {
        document.getElementById('Output').value = response.choices[0].message.content;
      },
      error: function(xhr, status, error) {
        console.error('에러 발생:', error);
        document.getElementById('Output').value = '오류가 발생했습니다.';
      }
    });
  }