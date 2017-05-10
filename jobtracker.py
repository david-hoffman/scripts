import os
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


def get_qstat(user="hoffmand"):
    """Get qstat for user"""
    process = subprocess.run(
        ["qstat", "-u", user],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if process.stderr:
        raise RuntimeError(process.stderr.decode())
    return process.stdout.decode().split(os.linesep)


def get_jobnames(qstat):
    """Extract job names from get_qstat return"""
    # The first line is header, the second line is dashes
    # and the last line is just a return \n
    return [q.split()[2] for q in qstat[2:-1]]


def get_status(user, key):
    """Return true if any job matching key is returned"""
    qstat = get_qstat(user)
    jobs = get_jobnames(qstat)
    return any(key in job for job in jobs)


def changed(old, new):
    """Test if there's a change in status

    Return new status and True if status has changed
    or False if status is the same"""
    return new, old is not new


def watcher(jobkey, user, recipient, subject, poletime):
    """Set a watcher for qsub jobs that will send an email alert"""
    old_status = False
    while True:
        # dave_stat = !qstat -j Group* -u hoffmand
        new_status = get_status(user, jobkey)
        old_status, status_changed = changed(old_status, new_status)
        if status_changed:
            if not old_status:
                send_job_done(recipient, subject)
                click.echo("Job done")
            else:
                click.echo("Job started")
        time.sleep(poletime)


@click.command()
@click.option('--jobkey', default="Group", help="Argument to pass to qsub's -j option")
@click.option('--user', default=None, help="The user who's jobs are running")
@click.option('--recipient', default=None, help="The recipient's email address")
@click.option('--subject', default=None, help="The subject line of the email")
@click.option('--poletime', default=60, help="Time between poles for (in seconds)")
def cli(jobkey, user, recipient, subject, poletime):
    """The thing that does work"""
    if subject is None:
        subject = "qsub {} Job Done".format(jobkey)
    if user is None:
        user = os.getlogin()
    if recipient is None:
        recipient = "{}@janelia.hhmi.org".format(user)
    click.echo("Watching for jobs '{}' for user {}".format(jobkey, user))
    watcher(jobkey, user, recipient, subject, poletime)


if __name__ == '__main__':
    cli()
