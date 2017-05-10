import time
import subprocess
import smtplib
import click
from email.mime.text import MIMEText


def send_job_done(recipient, sub):
    """Send an email alert"""
    msg = MIMEText("Job Done")
    msg["Subject"] = sub
    msg["From"] = "qsub-alert@janelia.hhmi.org"
    msg["To"] = recipient
    s = smtplib.SMTP("localhost")
    s.send_message(msg)
    s.quit()


def get_status(user, key):
    """emulate the following ipython call:
    dave_stat = !qstat -j Group* -u hoffmand
    """
    process = subprocess.run(
        ["qstat", "-j", key, "-u", user],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if process.stderr:
        raise RuntimeError(process.stderr.decode())
    status = process.stdout.decode().split("\n")
    return status


def changed(old, new):
    """Test if there's a change in status

    Return new status and True if status has changed
    or False if status is the same"""
    return new, old is not new


@click.command()
@click.option('--jobkey', default="Group*", help="Argument to pass to qsub's -j option")
@click.option('--user', default="hoffmand", help="The user who's jobs are running")
@click.option('--recipient', default="hoffmand@janelia.hhmi.org", help="The recipient's email address")
@click.option('--subject', default="qsub Job Done", help="The subject line of the email")
def watcher(jobkey, user, recipient, subject):
    """Set a watcher for qsub jobs that will send an email alert"""
    tester = 'Following jobs do not exist or permissions are not sufficient: '
    old_status = False
    while True:
        # dave_stat = !qstat -j Group* -u hoffmand
        new_status = get_status(user=user, key=jobkey)
        old_status, status_changed = changed(old_status, tester not in new_status)
        if status_changed:
            if not old_status:
                send_job_done(recipient, subject)
                print("Job done")
            else:
                print("Job started")
        time.sleep(60 * 5)


if __name__ == '__main__':
    watcher()
