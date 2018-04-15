from flask_wtf import FlaskForm
from wtforms import (TextField, StringField, PasswordField, TextAreaField,
                     IntegerField, BooleanField,
                     SubmitField, FloatField, SelectField)
from wtforms.validators import (DataRequired, Regexp, ValidationError, Email,
                                Length, EqualTo)

class AdminForm(FlaskForm):
    stream_type = StringField("Youtube or Naver?", validators = [DataRequired()])
